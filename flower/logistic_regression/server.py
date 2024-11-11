import os
from typing import Dict
import argparse
import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import utils
from sklearn.metrics import log_loss
# import wandb


# Load your dataset
df = pd.read_csv("x_one_complete.csv")


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression, num_clients:int, test_split=0.2, random_seed=42):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, X_test, _, y_test = utils.load_data(df, random_seed=42, test_split=0.2)

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        results_directory = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/plots/fed' 
        results_file = os.path.join(results_directory, 'results.csv')
        results = pd.read_csv(results_file)

        # Update model with th[e latest parameters
        utils.set_model_params(model, parameters)
        y_pred = model.predict(X_test)
        loss = log_loss(y_test, model.predict_proba(X_test))
        scores = utils.get_scores(y_test, y_pred)
        model_info = {'Model Name': 'Logistic Regression', 'Loss': loss, 'Accuracy': scores['accuracy']}
        new_row = pd.DataFrame([model_info])
        results = pd.concat([results, new_row], ignore_index=True)
        results.to_csv(results_file, index=False)
       
        print(f"\nServer accuracy: {scores['accuracy']}")         
        return loss, {"accuracy": scores["accuracy"]}

    return evaluate

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
    
    #create a model
    model = LogisticRegression()
    model = utils.set_initial_params(model)

    #define the strategy
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model,  num_clients=num_clients),
        on_fit_config_fn=fit_round,
        )

    #start the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Listening on all interfaces, port 8080
        config=fl.server.ServerConfig(num_rounds=3),  # Number of training rounds
        strategy=strategy,
    )

    
 
    # model_info = {
    #     'Model Name': row['algoritmos'],
    #     'Loss': row['log_loss'],
    #     'Accuracy': row['accuracy'],

    # }