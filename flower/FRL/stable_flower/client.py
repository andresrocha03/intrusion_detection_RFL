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
        return utils.get_weights(self.model)

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        utils.set_weights(self.model, parameters)
        self.model.learn(total_timesteps=1000, reset_num_timesteps=False)
        return utils.get_weights(self.model), len(self.x_train) , {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        utils.set_weights(self.model, parameters)
        # mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=1000)
        vec_env = self.model.get_env()
        obs = vec_env.reset()
        predictions = []
        attempts, correct = 0, 0      
        for i in range(self.x_test.shape[0]):
            action, _states = self.model.predict(obs)
            predictions.append(action[0])
            obs, rewards, dones, info = vec_env.step(action)
            attempts += 1
            if rewards > 0:
                correct += 1
        loss = log_loss(self.y_test, predictions)
        accuracy = (predictions == self.y_test).mean()
        return loss, len(self.x_test), {"accuracy": accuracy}

def create_client(cid: str):
    #get train and test data
    X_train, X_test, y_train, y_test = utils.load_data(partitions[int(cid)-1], random_seed=42, test_split=0.2)
    env = TabularEnv((X_train, y_train), row_per_episode=1, random=False)
    env.reset()
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