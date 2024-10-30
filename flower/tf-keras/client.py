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
df = pd.read_csv("x_one_complete.csv")

# Define Flower Client
class SimpleClient(NumPyClient):
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=10,
        batch_size=32,
        verbose=0,
    ):
        self.model = utils.load_model()
        self.x_train, self.y_train, self.x_test, self.y_test = X_train, y_train, X_test, y_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
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
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}


def create_client(cid: str):
    #get train and test data
    X_train, X_test, y_train, y_test = utils.load_data(partitions[int(cid)-1], random_seed=42, test_split=0.2)
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
    partitions = utils.partition_data(df, args.num_clients)


    # Assuming the partitioner is already set up elsewhere and loaded here
    fl.client.start_client(server_address="0.0.0.0:8080", client=create_client(args.id).to_client())