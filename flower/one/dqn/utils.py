from collections import OrderedDict
import torch
from typing import List
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



NUM_UNIQUE_LABELS = 2  # Number of unique labels in your dataset
NUM_FEATURES = 15  # Number of features in your dataset

def load_data(partition: list[NDArray]):
    """Load data."""
    X = partition.drop('label', axis=1).values
    y = partition['label'].values
    return X, y 

def partition_data(data, num_partitions):
# Partitioning the dataset into parts for each client
    return np.array_split(data, num_partitions)

def get_weights(model):
    model_weights = {}
    all_model_weights = model.get_parameters()['policy']
    model_weights = {}
    #get only the training net parameters -> the ones without "target" in the key
    for key, val in all_model_weights.items():
        if "target" not in key:
            model_weights[key] = val
    return [val.cpu().numpy() for _, val in model_weights.items()]


def set_weights(model, parameters):
    #copy the parameters not related to the training net
    new_params = model.get_parameters()

    #select parameters from the training net
    param_names = []
    for key, value in new_params['policy'].items():
        if "target" not in key:
            param_names.append(key)
    
    #adjust new parameters format to the model
    params_dict = zip(param_names, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    
    #copy the state dict to the new params
    for name in param_names:
        new_params['policy'][name] = state_dict[name]
    model.set_parameters(new_params)

