from typing import Dict, List, Tuple
import argparse
import flwr as fl
import numpy as np
import pandas as pd
import utils
# import wandb
from flwr.common import Context, Metrics, ndarrays_to_parameters


# Load your dataset
df = pd.read_csv("x_one_complete.csv")


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}



# # Define metric aggregation function
# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}



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
    
    parameters = ndarrays_to_parameters(utils.load_model().get_weights())
    print(parameters)

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
