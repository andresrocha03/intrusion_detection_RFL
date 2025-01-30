import sys
import os
# adding environment to the system path
sys.path.insert(0, '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/environments')
import pandas as pd
import warnings
from flwr.client import NumPyClient
import flwr as fl
from sklearn.metrics import log_loss
import argparse
import utils
from stable_baselines3 import DQN
import gymnasium as gym
from tabularenv import TabularEnv
import numpy as np

warnings.filterwarnings("ignore")

# Load your dataset
data_folder = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/processed_data/new_try'
df_train, df_test = utils.load_dataset(data_folder)

# Define Flower Client
class SimpleClient(NumPyClient):
    def __init__(self, cid, env_train, env_test, X_train, y_train, X_test, y_test):
        self.model_train = DQN("MlpPolicy", env_train)
        self.env_train = env_train
       
        self.model_test = DQN("MlpPolicy", env_test)
        self.env_test = env_test
        
        self.x_train, self.y_train, self.x_test, self.y_test = X_train, y_train, X_test, y_test
        self.client_id = cid
    
    def get_parameters(self, config):
        return utils.get_weights(self.model_train)

    def fit(self, parameters, config):
        """Train the model with data of this client."""

        utils.set_weights(self.model_train, parameters)
        
        print(f"before training client {self.client_id}: {self.env_train.dataset_idx}")
        self.model_train.learn(total_timesteps=(0.5*len(self.x_train)), reset_num_timesteps=False, progress_bar=True)        
        print(f"after training client {self.client_id}: {self.env_train.dataset_idx}")
        print(f"Training model for client {self.client_id} finished")
        print(self.env_train.info)
        print("------------------------------------------------------------")

        return utils.get_weights(self.model_train), len(self.x_train) , {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""

        #carregar modelo global
        utils.set_weights(self.model_train, parameters)
               
        self.model_test.set_parameters(self.model_train.get_parameters())
                
        obs, info = self.env_test.reset(seed=0)
        predictions = []

        for i in range(self.y_test.shape[0]):
            action, _states = self.model_train.predict(obs)
            predictions.append(action)
            obs, rewards, terminated, truncated, info = self.env_test.step(action)
            if terminated:
                self.env_test.reset()   
        
        prob_predictions = utils.sigmoid(np.asarray(predictions))
        loss = log_loss(self.y_test, prob_predictions)
        accuracy = (predictions == self.y_test ).mean()
        print(f"TESTING loss and accuracy for client {self.client_id}: {loss} and {accuracy}\n")

        return loss, len(self.x_test), {"loss":loss, "accuracy": accuracy}

def create_client(cid: str):
    #get train and test data
    X_train, y_train  = utils.load_client_data(train_partitions[int(cid)-1])
    X_test, y_test = utils.load_client_data(test_partitions[int(cid)-1])

    env_train = TabularEnv(X_train, y_train)
    
    env_test = TabularEnv(X_test, y_test)
    
    return SimpleClient(int(cid), env_train, env_test, X_train, y_train, X_test, y_test)

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