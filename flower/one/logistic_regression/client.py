import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
import numpy as np
from flwr.client import NumPyClient
import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import argparse
import utils

warnings.filterwarnings("ignore")

# Load your dataset
data_folder = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/processed_data/new_try'
df_train, df_test = utils.load_dataset(data_folder)

class SimpleClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.model = LogisticRegression(penalty='l1',warm_start=True, solver="saga", max_iter=1)
        # Setting initial parameters, akin to model.compile for keras models
        utils.set_initial_params(self.model)


    def get_parameters(self, config):
        return utils.get_model_parameters(self.model)
    
    def set_parameters(self, parameters):
       return utils.set_model_params(self.model, parameters)

    def fit(self, parameters, config):
        # print(f"Client {args.id} model parameters: {self.get_parameters(config)}")
        # print(f"Client {args.id} received parameters: {parameters}")
        # print(f"Client {args.id} updated parameters {self.get_parameters(config)}")
        # print(f"Client {args.id} fitted model parameters {self.get_parameters(config)}")
       
        self.model = self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
            
        return utils.get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # print(f"parameters for client {args.id}: {parameters}")

        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        print(f"Client accuracy for client {args.id}: {accuracy} ")
        return loss, len(self.X_test), {"loss": loss, "accuracy": accuracy}


def create_client(cid: str):
    #get train and test data
    X_train, y_train = utils.load_train_data(train_partitions[int(cid)-1])
    X_test, y_test = utils.load_test(test_partitions[int(cid)-1])
    return SimpleClient(X_train, y_train, X_test, y_test)

if __name__ == "__main__":

    # Parse command line arguments for the partition ID
    parser = argparse.ArgumentParser(description="Flower client using a specific data partition")
    parser.add_argument("--id", type=int, required=True, help="Data partition ID")
    
    #parse number of clients
    parser.add_argument("--num_clients",type=int,required=True,
                        help="Specifies how many clients the bash script will start.")

    args = parser.parse_args()
    # partition the data
    train_partitions = utils.partition_data(df_train, args.num_clients)
    test_partitions = utils.partition_data(df_test, args.num_clients)

    # Assuming the partitioner is already set up elsewhere and loaded here
    fl.client.start_client(server_address="0.0.0.0:8080", client=create_client(args.id).to_client())