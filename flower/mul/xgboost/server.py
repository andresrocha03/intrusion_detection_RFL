import os
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
    """Return an aggregated metric (AUC, Loss) for evaluation."""
    results_directory = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/plots/fed/mul' 
    results_file = os.path.join(results_directory, 'xgb_res.csv')
    results = pd.read_csv(results_file)


    total_num = sum([num for num, _ in eval_metrics])

    loss_aggregated = (
        sum([metrics["Loss"] * num for num, metrics in eval_metrics]) / total_num
    )
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    acc_aggregated = (
       sum([metrics["Accuracy"] * num for num, metrics in eval_metrics]) / total_num
    )

    metrics_aggregated = {"AUC": auc_aggregated,
                          "Loss": loss_aggregated,
                          "Accuracy": acc_aggregated}
    
    model_info = {'Model Name': 'XGBoost', 
                  'Loss': loss_aggregated, 'Accuracy': acc_aggregated}
    new_row = pd.DataFrame([model_info])
    results = pd.concat([results, new_row], ignore_index=True)
    results.to_csv(results_file, index=False)
    
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
        config=fl.server.ServerConfig(num_rounds=10),  # Number of training rounds
        strategy=strategy,
    )

