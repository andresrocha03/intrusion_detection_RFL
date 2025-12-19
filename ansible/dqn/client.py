import sys
import os
# adding environment to the system path
sys.path.insert(0, '/home/pi/fed_project/environments')
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
from scipy.special import softmax

warnings.filterwarnings("ignore")

# Load your dataset
data_folder = '/home/pi/fed_project/data/'
df_train, df_test = utils.load_dataset(data_folder)
# df_surprise = utils.load_surprise_dataset(data_folder)

# Define Flower Client
class SimpleClient(NumPyClient):
    def __init__(self, cid, env_train, env_test, X_train, y_train, X_test, y_test, model_name, lr_param, gm, seed):        
        # lr = pow(10, -lr_param)
        # gm = pow(10, -gm)
        # lr *= gm
        # gm = 3e-1
        # print(f"Learning rate: {lr} and gamma: {gm}")

        self.model_train = DQN("MlpPolicy", env_train, learning_rate=3e-5, seed=42, tensorboard_log=f"logs/")
        self.env_train = env_train
       
        self.model_test = DQN("MlpPolicy", env_test)
        self.env_test = env_test
        
        self.x_train, self.y_train, self.x_test, self.y_test = X_train, y_train, X_test, y_test
        self.client_id = cid

        self.data_fraction = int(0.7*len(self.x_train))

        self.round = 0

        self.model_name = model_name

        self.seed = seed

    def get_parameters(self, config):
        return utils.get_weights(self.model_train)

    def fit(self, parameters, config):
        """Train the model with data of this client."""

        utils.set_weights(self.model_train, parameters)
        logdir = f"{self.model_name}_final_verdade"
        self.model_train.learn(total_timesteps=self.data_fraction, reset_num_timesteps=False, progress_bar=True, tb_log_name=logdir)        
        self.round += 1
        return utils.get_weights(self.model_train), self.data_fraction , {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""

        #carregar modelo global
        utils.set_weights(self.model_train, parameters)
               
        self.model_test.set_parameters(self.model_train.get_parameters())
        
        
        obs, info = self.env_test.reset(seed=0)
        predictions = []
        prob_predictions = []

        # if self.round == 101 or self.round == 100:
        #     #save model
        #     self.model_train.save(f"models/{self.model_name}_{self.seed}") 
        #     env_surprise = TabularEnv(df_surprise.drop('label', axis=1).values, df_surprise['label'].values)
        #     obs, info = env_surprise.reset(seed=0)
        #     self.model_test = DQN("MlpPolicy", env_surprise)
        #     self.model_test.set_parameters(self.model_train.get_parameters())
        #     self.env_test = env_surprise
        #     self.y_test = df_surprise['label'].values

        for i in range(self.y_test.shape[0]):
            action, _states = self.model_test.predict(obs)
            #extract q values
            probabilities = self.model_test.q_net.q_values_aux
            
            #transform tensor to numpy
            probabilities = probabilities.detach().numpy()

            #normalize q values
            probabilities = probabilities / np.max(probabilities)
            
            #calculate the probability of each action
            # print(probabilities)
            probabilities = softmax(probabilities)
            # print(probabilities)
            #reshape for 1D
            probabilities = probabilities.reshape(-1)
            
            predictions.append(action)
            prob_predictions.append(probabilities)

            obs, rewards, terminated, truncated, info = self.env_test.step(action)
            if terminated:
                self.env_test.reset()   
        
        
        
        # prob_predictions = utils.sigmoid(np.asarray(predictions))
        loss = log_loss(self.y_test, prob_predictions)
        accuracy = (predictions == self.y_test ).mean()
        print(f"TESTING loss and accuracy for client {self.client_id}: {loss} and {accuracy}\n, round {self.round}")

        return loss, len(self.x_test), {"loss":loss, "accuracy": accuracy, "precision": info['precision'], "recall": info['recall']}

def create_client(cid: str, lr: int, gm: int, seed: int) -> SimpleClient:
    #get train and test data
    X_train, y_train  = utils.load_client_data(train_partitions[int(cid)-1])
    X_test, y_test = utils.load_client_data(test_partitions[int(cid)-1])

    env_train = TabularEnv(X_train, y_train)
    
    env_test = TabularEnv(X_test, y_test)

    #choose model name
    model_name = f"Client_{cid}"

    return SimpleClient(int(cid), env_train, env_test, X_train, y_train, X_test, y_test, model_name, lr, gm, seed)

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

    # #parse learning rate
    # parser.add_argument("--learning_rate",type=int,required=True,
    #                     help="Specifies learning rate power.")

    # #parse discount factor
    # parser.add_argument("--gamma",type=int,required=True,
    #                     help="Specifies gamma power.")

    # #parse discount factor
    # parser.add_argument("--seed",type=int,required=True,
    #                     help="Specifies gamma power.")


    args = parser.parse_args()
   
    # partition the data
    train_partitions = utils.partition_data(df_train, args.num_clients)
    test_partitions = utils.partition_data(df_test, args.num_clients)
    
    # Assuming the partitioner is already set up elsewhere and loaded here
    # fl.client.start_client(server_address="0.0.0.0:8080", client=create_client(args.id, args.learning_rate, args.gamma, args.seed).to_client())      
   
    # após subir o servidor, execute "hostname -i | awk '{print $1}'" na sua maquina para obter o endereço IP do servidor
    fl.client.start_client(server_address="10.10.10.237:8080", client=create_client(args.id, lr=3e-5, gm=0.99, seed=42).to_client())   
