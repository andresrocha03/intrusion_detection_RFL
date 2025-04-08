import sys
# sys.path.insert(0, '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/environments')

import os
# from tabularenv import TabularEnv
from typing import Dict, List, Tuple
import argparse
import flwr as fl
import numpy as np
import pandas as pd
import utils
from stable_baselines3 import DQN
from flwr.common import Context, Metrics, ndarrays_to_parameters



def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    results_directory = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/results/fed/mul' 
    results_file = os.path.join(results_directory, 'dqn_res.csv')
    results = pd.read_csv(results_file)
    
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    acc_aggregated = sum(accuracies) / sum(examples)

    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    loss_aggregated = sum(losses) / sum(examples)

    metrics_aggregated = {"Loss": loss_aggregated,
                          "Accuracy": acc_aggregated}
    
    model_info = {'Model Name': 'DQN', 
                  'Loss': loss_aggregated, 'Accuracy': acc_aggregated}
    new_row = pd.DataFrame([model_info])
    results = pd.concat([results, new_row], ignore_index=True)
    results.to_csv(results_file, index=False)

    # Aggregate and return custom metric (weighted average)
    return metrics_aggregated

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
    
    #load env
        # Load your dataset
    data_folder = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/processed_data/new_try'
    # df_train, df_test = utils.load_dataset(data_folder)
    # X_test, y_test = utils.load_client_data(df_test)
    # env = TabularEnv(X_test, y_test)

    #initital parameters
    # parameters = utils.get_weights(DQN("MlpPolicy", env))


    #define the strategy
    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        min_available_clients=2,
        on_fit_config_fn=fit_round,
        # initial_parameters=parameters,
        )

    #start the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Listening on all interfaces, port 8080
        config=fl.server.ServerConfig(num_rounds=10),  # Number of training rounds
        strategy=strategy
    )