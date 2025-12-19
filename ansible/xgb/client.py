import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from flwr.client import Client
import flwr as fl
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score
import argparse
import utils
import xgboost as xgb
from flwr.common import GetParametersIns, GetParametersRes,Parameters, Status, Code, FitRes,EvaluateIns,EvaluateRes



warnings.filterwarnings("ignore")


# Load your dataset
data_folder = '/home/pi/fed_project/data/'
df_train, df_test = utils.load_dataset(data_folder)
# df_surprise = utils.load_surprise_dataset(data_folder)
# surprise, len_sur = utils.load_data(df_surprise)


# Setting initial parameters, akin to model.compile for keras models
#set parameters to prevent overfitting
params = {
    "objective": "binary:logistic",
    "eta": 0.05,
    "max_depth": 2,
    "min_child_weight": 1,
    "gamma": 0.05   ,
    "subsample": 1,
    "colsample_bytree": 1,
    "lambda": 1,
}

class SimpleClient(Client):
    def __init__(self, train, test,len_train,len_test,num_local_round,params):
        self.first = None
        self.config = None
        self.model = None
        self.train = train
        self.test = test
        self.len_train = len_train
        self.len_test = len_test
        self.num_local_round = num_local_round
        self.params = params

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self,ins)
        return GetParametersRes(status=Status(code=Code.OK,message="OK"),
                                parameters=Parameters(tensor_type="",tensors=[])
                                )
    def _local_boost(self, model_input):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            model_input.update(self.train, model_input.num_boosted_rounds())
        
        # Bagging: extract the last N=num_local_round trees for sever aggregation
        output_model = model_input[
            model_input.num_boosted_rounds()
            - self.num_local_round : model_input.num_boosted_rounds()
        ]
        self.num_local_round += 1
        
        return output_model
    
    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        if not self.first:
            #first round
            self.first = True
            model = xgb.train(
                                self.params, 
                                self.train,
                                num_boost_round=self.num_local_round,
                                evals=[(self.test, "test"), (self.train, "train")],
                            )
            
            self.config = model.save_config()
            # self.model = model
        else:
            model = xgb.Booster(params=self.params)
            global_model = bytearray(ins.parameters.tensors[0])
            
            #load global model
            model.load_model(global_model)
            model.load_config(self.config)
            
            #local_training
            model = self._local_boost(model)
        
        #save model
        local_model = model.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
                status=Status(
                    code=Code.OK,
                    message="OK",
                ),
                parameters=Parameters(
                    tensor_type="",
                    tensors=[local_model_bytes],
                ),
                num_examples=self.len_train,
                metrics={},
        )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        model = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        model.load_model(para_b)
        
        # if self.num_local_round == 101:
        #     self.test = surprise
        #     print("Testing on surprise dataset") 

        # Run evaluation
        eval_results = model.eval_set(
            evals=[(self.test, "test")],
            iteration=model.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
        
     

        probabilities = model.predict(self.test)
        predictions = np.where(probabilities > 0.5, 1, 0)        #convert probabilities to binary
        loss = log_loss(self.test.get_label(), probabilities)
        accuracy = accuracy_score(self.test.get_label(), predictions)
        precision = precision_score(self.test.get_label(), predictions)
        recall = recall_score(self.test.get_label(), predictions)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=loss,
            num_examples=self.len_test,
            metrics={"Loss": loss,
                     "AUC": auc,
                     "Accuracy": accuracy,
                     "Precision": precision,
                     "Recall": recall},
        )

 

def create_client(cid: str):
    #get train and test data
    train, len_train = utils.load_data(train_partitions[int(cid)-1])
    test, len_test = utils.load_data(test_partitions[int(cid)-1])
    num_local_round = 1
    return SimpleClient(train, test, len_train, len_test, num_local_round, params)


if __name__ == "__main__":

    # Parse command line arguments for the partition ID
    parser = argparse.ArgumentParser(description="Flower client using a specific data partition")
    parser.add_argument("--id", type=int, required=True, help="Data partition ID")
    
    #parse number of clients
    parser.add_argument("--num_clients",type=int,required=True,
                        help="Specifies how many clients the bash script will start.")
    #parse number of rounds
    parser.add_argument("--num_rounds",type=int,required=True,
                        help="Specifies how many rounds the bash script will run.")


    args = parser.parse_args()

    # partition the data
    train_partitions = utils.partition_data(df_train, args.num_clients)       
    test_partitions = utils.partition_data(df_test, args.num_clients)    

    # após subir o servidor, execute "hostname -i | awk '{print $1}'" na sua maquina para obter o endereço IP do servidor

    # Assuming the partitioner is already set up elsewhere and loaded here
    fl.client.start_client(server_address="10.10.10.237:8080", client=create_client(args.id))