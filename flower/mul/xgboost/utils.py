import os
from typing import List
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb

def load_dataset(data_folder: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def load_data(partition: list[NDArray]):
    """Load data."""
    X = partition.drop('label', axis=1).values
    y = partition['label'].values
    data = xgb.DMatrix(X, label=y)
    return data, len(X) 

def partition_data(data, num_partitions):
# Partitioning the dataset into parts for each client
    return np.array_split(data, num_partitions)
