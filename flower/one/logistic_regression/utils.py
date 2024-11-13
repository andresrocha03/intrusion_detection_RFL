from typing import List
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

NUM_UNIQUE_LABELS = 2  # Number of unique labels in your dataset
NUM_FEATURES = 15  # Number of features in your dataset

def load_data(partition: list[NDArray], test_split: float, random_seed=42):
    """Load data."""
    X = partition.drop('label', axis=1).values
    y = partition['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_seed)
    return X_train, X_test, y_train, y_test 

def partition_data(data, num_partitions):
# Partitioning the dataset into parts for each client
    return np.array_split(data, num_partitions)

def set_initial_params(model: LogisticRegression):
    """Set initial parameters for the model."""
    model.classes_ = np.array([i for i in range(NUM_UNIQUE_LABELS)])
    model.coef_ = np.zeros((NUM_UNIQUE_LABELS, NUM_FEATURES))
    if model.fit_intercept:
        model.intercept_ = np.zeros((NUM_UNIQUE_LABELS,))
    return model

def set_model_params(model: LogisticRegression, params: List[NDArray]):
    """Set model parameters."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def get_model_parameters(model: LogisticRegression):
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params

def get_scores(y_true: NDArray, y_pred: NDArray) -> dict:
    score = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
    }
    return score