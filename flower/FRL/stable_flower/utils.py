from collections import OrderedDict
import torch
from typing import List
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



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

def get_weights(model):
    model_weights = {}
    model_weights = model.get_parameters()['policy']
    print(model.get_parameters()['policy.optimizer'])
    return [val.cpu().numpy() for _, val in model_weights.items()]


def set_weights(model, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)