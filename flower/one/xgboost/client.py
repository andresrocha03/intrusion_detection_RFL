import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from flwr.client import Client
import flwr as fl
from sklearn.metrics import log_loss, accuracy_score
import argparse
import utils
import xgboost as xgb
from flwr.common import GetParametersIns, GetParametersRes,Parameters, Status, Code, FitRes,EvaluateIns,EvaluateRes



warnings.filterwarnings("ignore")


# Load your dataset
df = pd.read_csv("x_one_complete.csv")


model = xgb.XGBClassifier()

# Setting initial parameters, akin to model.compile for keras models

num_local_round = 1
params = {
    "objective": "binary:logistic",
    "eta": 0.1,
    "max_depth": 8,
    "eval_metric": "auc",
    "nthread": 16,
    "num_paralell_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
}


class SimpleClient(Client):
    def __init__(self, train, test,num_train,num_test,num_local_round,params):
        self.model = None
        self.config = None
        self.train = train
        self.test = test
        self.num_train = num_train
        self.num_test = num_test
        self.num_local_round = num_local_round
        self.params = params

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self,ins)
        return GetParametersRes(status=Status(code=Code.OK,message="OK"),
                                parameters=Parameters(tensor_type="",tensors=[])
                                )
    
    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        if not self.model:
            #first round
            model = xgb.train(params, 
                                   self.train,
                                   num_boost_round=num_local_round,
                                   evals=[(self.test, "test"), (self.train, "train")],)
            self.config = model.save_config()
            self.model = model
        else:
            for item in ins.parameters.tensors:
                global_model = bytearray(item)
            
            self.model.load_model(global_model)
            self.model.load_config(self.config)
            
            model = self._local_boost()
        
        local_model = self.model.save_raw("json")
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
                num_examples=self.num_train,
                metrics={},
        )

    def _local_boost(self):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            self.model.update(self.train, self.model.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        model = self.model[
            self.model.num_boosted_rounds()
            - self.num_local_round : self.model.num_boosted_rounds()
        ]

        return model
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Run evaluation
        eval_results = self.model.eval_set(
            evals=[(self.test, "test")],
            iteration=self.model.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
        loss = log_loss(self.test.get_label(), self.model.predict(self.test))
        #convert probabilities to binary
        
        predictions = self.model.predict(self.test)
        predictions = np.where(predictions > 0.5, 1, 0)
        accuracy = accuracy_score(self.test.get_label(), predictions)
        
        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=loss,
            num_examples=self.num_test,
            metrics={"Loss": loss,
                     "AUC": auc,
                     "Accuracy": accuracy},
        )

 

def create_client(cid: str):
    #get train and test data
    train, test, num_train, num_test = utils.load_data(partitions[int(cid)-1], random_seed=42, test_split=0.2)
    return SimpleClient(train, test, num_train, num_test, num_local_round, params)


if __name__ == "__main__":

    # Parse command line arguments for the partition ID
    parser = argparse.ArgumentParser(description="Flower client using a specific data partition")
    parser.add_argument("--id", type=int, required=True, help="Data partition ID")
    
    #parse number of clients
    parser.add_argument("--num_clients",type=int,required=True,
                        help="Specifies how many clients the bash script will start.")

    args = parser.parse_args()

    # partition the data
    partitions = utils.partition_data(df, args.num_clients)    
   
    # Assuming the partitioner is already set up elsewhere and loaded here
    fl.client.start_client(server_address="0.0.0.0:8080", client=create_client(args.id))