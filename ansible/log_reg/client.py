#!/usr/bin/env python3

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
data_folder = '/home/pi/fed_project/data/'
df_train, df_test = utils.load_dataset(data_folder)
# df_surprise = utils.load_surprise_dataset(data_folder)

class SimpleClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test):
        #exclude part of the data for training
        X_train = X_train[:int(len(X_train))]
        y_train = y_train[:int(len(y_train))]
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.model = LogisticRegression(penalty='l1',warm_start=True, solver="saga", max_iter=1)
        self.round = 0
        # Setting initial parameters, akin to model.compile for keras models
        utils.set_initial_params(self.model)

    def get_parameters(self, config):
        return utils.get_model_parameters(self.model)
    
    def set_parameters(self, parameters):
        return utils.set_model_params(self.model, parameters)

    def fit(self, parameters, config): 
        self.model = self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        self.round += 1
        return utils.get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model = self.set_parameters(parameters)

        # if self.round == 101:
        #     #save model
        #     self.X_test = df_surprise.drop('label', axis=1).values
        #     self.y_test = df_surprise['label'].values
        #     print("Testing on surprise dataset")

        y_prob = self.model.predict_proba(self.X_test)
        scores = utils.get_scores(self.y_test, y_prob)
        print(f"Client loss, accuracy, precision and recall for client {args.id}:{scores['loss']}, {scores['accuracy']}, {scores['precision']}, {scores['recall']}")
        return scores["loss"], len(self.X_test), scores


def create_client(cid: str):
    #get train and test data
    X_train, y_train = utils.load_train_data(train_partitions[int(cid)-1])
    X_test, y_test = utils.load_test(test_partitions[int(cid)-1])
    return SimpleClient(X_train, y_train, X_test, y_test)

if __name__ == "__main__":

    # Parse command line arguments for the partition ID
    parser = argparse.ArgumentParser(description="Flower client using a specific data partition")
    
    #client id
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
    fl.client.start_client(server_address="10.10.10.237:8080", client=create_client(args.id).to_client())