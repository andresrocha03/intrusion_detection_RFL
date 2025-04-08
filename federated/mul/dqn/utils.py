import os
from collections import OrderedDict
import string
from stable_baselines3 import DQN
import torch
from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


NUM_UNIQUE_LABELS = 9  # Number of unique labels in your dataset
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
    X_train = pd.read_csv(os.path.join(data_folder, "x_mul_train.csv" ))
    y_train = pd.read_csv(os.path.join(data_folder, "y_mul_train.csv"))
    X_train['label'] = y_train
    df_train = X_train

    X_test = pd.read_csv(os.path.join(data_folder, "x_mul_test.csv"))
    y_test = pd.read_csv(os.path.join(data_folder, "y_mul_test.csv"))
    X_test['label'] = y_test
    df_test = X_test

    return df_train, df_test 

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
    return 1 / (1 + np.exp(-x))

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

def predict_probabilities(model: DQN, obs: np.ndarray)-> np.ndarray:
    """
    Predicts the probabilities of each action for a given observation.
    
    Parameters:
    - model (DQN): The model to predict.
    - obs (np.ndarray): The observation to predict the probabilities for.

    Returns:
    - probabilities (np.ndarray): The probabilities of each action.
    """
    #predict 100 times for one observation
    predictions = [model.predict(obs)[0] for i in range(100)] 
  
    #count the number of times each action was predicted
    counts = np.bincount(predictions, minlength=9) 
    
    #calculate the probability of each action
    probabilities = counts/100 
    
    return probabilities


def model_predict(model: DQN, obs: np.ndarray, predictions:np.ndarray) -> int | np.ndarray:
    """
    Predicts the action to take for a given observation.
    
    Parameters:
    - model (DQN): The model to predict.
    - obs (np.ndarray): The observation to predict the action for.
    - attack_type (string): The type of attack to predict the action for.

    Returns:
    - action (int): The action to take 
    - probabilities (np.ndarray): The probabilities of each action.
    """
    #predict the probabilities of each action
    probabilities = predict_probabilities(model, obs) 
    predictions.append(probabilities)

    #return the action with the highest probability
    return np.argmax(probabilities)
