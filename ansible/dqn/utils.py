import os
from collections import OrderedDict
import string
import torch
from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUM_UNIQUE_LABELS = 2  # Number of unique labels in your dataset
NUM_FEATURES = 15  # Number of features in your dataset

def load_dataset(data_folder: string) -> Tuple[NDArray, NDArray, NDArray, NDArray]:  
    """
    Load dataset.

    Parameters:
    - data_folder: str
        Path to the folder containing the dataset.
    
    Returns:
    - df_train: pd.DataFrame
        Training dataset.
    - df_test: pd.DataFrame
        Test dataset.
    """
    X_train = pd.read_csv(os.path.join(data_folder, "x_one_train.csv" ))
    y_train = pd.read_csv(os.path.join(data_folder, "y_one_train.csv"))
    X_train['label'] = y_train
    df_train = X_train

    X_test = pd.read_csv(os.path.join(data_folder, "x_one_test.csv"))
    y_test = pd.read_csv(os.path.join(data_folder, "y_one_test.csv"))
    X_test['label'] = y_test
    df_test = X_test

    return df_train, df_test 

def load_surprise_dataset(data_folder: string) -> Tuple[NDArray, NDArray, NDArray, NDArray]:  
    """
    Load surprise dataset.

    Parameters:
    - data_folder: str
        Path to the folder containing the dataset.
    
    Returns:
    - df_train: pd.DataFrame
        Training dataset.
    - df_test: pd.DataFrame
        Test dataset.
    """
    X_test = pd.read_csv(os.path.join(data_folder, "x_sur_test.csv"))
    y_test = pd.read_csv(os.path.join(data_folder, "y_sur_test.csv"))
    X_test['label'] = y_test
    df_test = X_test

    return df_test 


def load_client_data(partition: list[NDArray]):
    """
    Load data.
    
    Parameters:
    - partition: list[np.ndarray]
        Partition of the dataset.
    
    Returns:
    - X: np.ndarray
        Features.
    - y: np.ndarray
        Labels.
    """
    X = partition.drop('label', axis=1).values
    y = partition['label'].values
    return X, y 


# Define the sigmoid function for binary classification
def sigmoid(x):
    return np.asarray(1 / (1 + np.exp(-x)))

def partition_data(data, num_partitions):
# Partitioning the dataset into parts for each client
    return np.array_split(data, num_partitions)

def get_weights(model):

    all_model_weights = model.get_parameters()['policy']
    q_net_weights = {}

    #get only the training net parameters -> the ones without "target" in the key
    for key, val in all_model_weights.items():
        if "target" not in key:
            q_net_weights[key] = val

    return [val.cpu().numpy() for _, val in q_net_weights.items()]

def set_weights(model, parameters):
    #copy the parameters not related to the training net
    new_params = model.get_parameters()

    #select parameters from the training net
    q_net_params = []
    for key, value in new_params['policy'].items():
        if "target" not in key:
            q_net_params.append(key)
    
    #adjust new parameters format to the model
    params_dict = zip(q_net_params, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    #copy the state dict to the new params
    for name in q_net_params:
        new_params['policy'][name] = state_dict[name]
    
    model.set_parameters(new_params)
    
    return model