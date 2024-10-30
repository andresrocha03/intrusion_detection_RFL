from typing import Dict
import argparse
import flwr as fl
import numpy as np
import pandas as pd
import utils
from sklearn.metrics import log_loss
import wandb
from flwr.server.strategy import FedXgbBagging


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"AUC": auc_aggregated}
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

    # Define strategy
    strategy = FedXgbBagging(
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        min_available_clients=num_clients,
        min_fit_clients=num_clients,
    )

    #start the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Listening on all interfaces, port 8080
        config=fl.server.ServerConfig(num_rounds=3),  # Number of training rounds
        strategy=strategy,
    )

