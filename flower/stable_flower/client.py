import pandas as pd
import warnings
from flwr.client import NumPyClient
import flwr as fl
from sklearn.metrics import log_loss
import argparse
import utils
from stable_baselines3 import A2C, PPO, DQN
import gymnasium as gym
from tabularenv_train import TabularEnv
from stable_baselines3.common.evaluation import evaluate_policy


warnings.filterwarnings("ignore")

# Load your dataset
df = pd.read_csv("x_one_complete.csv")



# Define Flower Client
class SimpleClient(NumPyClient):
    def __init__(self, env, X_train, y_train, X_test, y_test):
        self.model = DQN("MlpPolicy", env)
        self.env = env
        self.x_train, self.y_train, self.x_test, self.y_test = X_train, y_train, X_test, y_test
    
    def get_parameters(self, config):
        return self.model.get_parameters()

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        self.model.set_parameters(parameters)
        self.model.learn(total_timesteps=1000, reset_num_timesteps=False)
        return self.model.get_parameters(), len(self.x_train) , {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        self.model.set_parameters(parameters)
        mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=1000)
        return mean_reward, len(self.x_test), {"accuracy": std_reward}


def create_client(cid: str):
    #get train and test data
    env.reset()
    X_train, X_test, y_train, y_test = utils.load_data(partitions[int(cid)-1], random_seed=42, test_split=0.2)
    env = TabularEnv((X_train, y_train), row_per_episode=1, random=False)
    return SimpleClient(env, X_train, y_train, X_test, y_test)

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