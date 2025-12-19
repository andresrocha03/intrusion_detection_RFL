import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
import numpy as np
from flwr.client import NumPyClient
import flwr as fl
from sklearn.metrics import log_loss
import argparse
import utils
import tensorflow as tf

warnings.filterwarnings("ignore")

# Load your dataset
data_folder = '/home/pi/fed_project/data/'
df_train, df_test = utils.load_dataset(data_folder)
# df_surprise = utils.load_surprise_dataset(data_folder)


# Define Flower Client
class SimpleClient(NumPyClient):
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=1,
        batch_size=32,
        verbose=1,
    ):
        self.model = utils.load_model()
        # reduce the size of the dataset for training
        X_train = X_train[:int(len(X_train) * 1)]
        y_train = y_train[:int(len(y_train) * 1)]
        
        self.x_train, self.y_train, self.x_test, self.y_test = X_train, y_train, X_test, y_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.round = 0
        

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        self.round += 1
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        self.model.set_weights(parameters)

        # if self.round == 101:
        #     self.x_test = df_surprise.drop("label", axis=1).values
        #     self.y_test = df_surprise["label"].values
        #     print("Testing on surprise dataset")

        y_pred = self.model.predict(self.x_test)
        scores = utils.get_scores(self.y_test, y_pred)
        print(type(scores['loss']))
        return scores['loss'], len(self.x_test), scores


def create_client(cid: str):
    #get train and test data
    X_train, y_train = utils.load_data(train_partitions[int(cid)-1])
    X_test, y_test = utils.load_data(test_partitions[int(cid)-1])
    return SimpleClient(X_train, y_train, X_test, y_test)

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
    fl.client.start_client(server_address="10.10.10.237:8080", client=create_client(args.id).to_client())