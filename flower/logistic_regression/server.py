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
    def evaluate(parameters: fl.common.NDArrays, config):
        # Update model with th[e latest parameters
        utils.set_model_params(model, parameters)
        y_pred = model.predict(X_test)
        loss = log_loss(y_test, model.predict_proba(X_test))
        scores = utils.get_scores(y_test, y_pred)
        scores["loss"] = loss
        # wandb.log(scores)
        # print("OIOIOIOIOIIOI")
        print(f"Server accuracy: {scores['accuracy']}")        
        
        # if server_round == 5 or server_round == 10: 
        #     feature_importances = model.coef_[0]
        #     # Print feature importances for the central model.
        #     for feature, importance in zip(X_test.columns, feature_importances):
        #         print(f"\n {feature}: {importance}")
        
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

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="federated_learning_ids",
    #     # track hyperparameters and run metadata
    #     config={
    #         "model": "Logistic Regression",
    #         "dataset": "Intrusion Detection System",
    #         "epochs": 3,
    #         "n_features": 15,
    #         "test_split":  0.2,
    #         "num_clients": num_clients,         
    #         "num_rounds": 3,
    #     },
    # )


    #define the strategy
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn( model, random_seed=42, test_split=0.2, num_clients=num_clients),
        on_fit_config_fn=fit_round,
        )

    #start the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Listening on all interfaces, port 8080
        config=fl.server.ServerConfig(num_rounds=10),  # Number of training rounds
    )
