import os
from typing import Dict
import argparse
import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import utils
from sklearn.metrics import log_loss
from typing import List, Tuple
from flwr.common import Context, Metrics


# Load your dataset
data_folder = '/home/andre/anstart/dados'
df_train, df_test = utils.load_dataset(data_folder)

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression, num_clients:int, test_split=0.2, random_seed=42):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    X, y = utils.load_test(df_test)

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
       
        # Update model with th[e latest parameters
        model1 = utils.set_model_params(model, parameters)
        y_pred = model1.predict(X)
        loss = log_loss(y, model1.predict_proba(X))
        scores = utils.get_scores(y, y_pred)
       
        print(f"\nServer accuracy: {scores['accuracy']}")         
        return loss, {"accuracy": scores["accuracy"]}

    return evaluate


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    results_directory = '/home/andre/anstart/dados' 
    results_file = os.path.join(results_directory, 'log_reg_res.csv')
    results = pd.read_csv(results_file)
    
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    acc_aggregated = sum(accuracies) / sum(examples)

    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    loss_aggregated = sum(losses) / sum(examples)

    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    precision_aggregated = sum(precisions) / sum(examples)

    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    recall_aggregated = sum(recalls) / sum(examples)
    
    metrics_aggregated = {"Loss": loss_aggregated,
                          "Accuracy": acc_aggregated,
                          "Precision": precision_aggregated,
                          "Recall": recall_aggregated}
    
    model_info = {'Model Name': 'Logistic Regression', 
                  'Loss': loss_aggregated, 
                  'Accuracy': acc_aggregated,
                  'Precision': precision_aggregated,
                  'Recall': recall_aggregated}
    
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
        choices=range(1, 31),
        required=True,
        help="Specifies how many clients the bash script will start.",
    )

    parser.add_argument(
        "--num_rounds",
        type=int,
        default=10,
        choices=range(1, 110),
        required=True,
        help="Specifies how many rounds the bash script will execute.",
    )
    args = parser.parse_args()
    
    #create a model
    model = LogisticRegression()
    model = utils.set_initial_params(model)

    #define the strategy
    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        min_available_clients=args.num_clients,
        # evaluate_fn=get_evaluate_fn(model,  num_clients=num_clients),
        on_fit_config_fn=fit_round,
        )

    #start the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Listening on all interfaces, port 8080
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),  # Number of training rounds
        strategy=strategy,
    )
