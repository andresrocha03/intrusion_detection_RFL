from typing import Dict, List, Tuple
import argparse
import flwr as fl
import numpy as np
import pandas as pd
import utils
from tabularenv_train import TabularEnv
from stable_baselines3 import DQN
from flwr.common import Context, Metrics, ndarrays_to_parameters


# Load your dataset
df = pd.read_csv("x_one_complete.csv")


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


if __name__ == "__main__":
    # Parse input to get number of clients
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--num_clients",
        type=int,
        default=5,
        choices=range(1, 11),
        required=True,
        help="Specifies how many clients the bash script will start.",
    )
    args = parser.parse_args()
    num_clients = args.num_clients

 
    X_train, X_test, y_train, y_test = utils.load_data(df, random_seed=42, test_split=0.2)
    env = TabularEnv((X_train, y_train), row_per_episode=1, random=False)
    model = DQN("MlpPolicy", env)

    parameters = ndarrays_to_parameters(utils.get_weights(model))
    
    #define the strategy
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        on_fit_config_fn=fit_round,
        initial_parameters=parameters,
        # evaluate_metrics_aggregation_fn=weighted_average,
        )

    #start the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Listening on all interfaces, port 8080
        config=fl.server.ServerConfig(num_rounds=10),  # Number of training rounds
    )
