from typing import List
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb



def load_data(partition: list[NDArray]):
    """Load data."""
    X = partition.drop('label', axis=1).values
    y = partition['label'].values
    data = xgb.DMatrix(X, label=y)
    return data, len(X) 

def partition_data(data, num_partitions):
# Partitioning the dataset into parts for each client
    return np.array_split(data, num_partitions)
