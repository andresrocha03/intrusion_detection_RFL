from typing import List
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


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


def load_model(learning_rate=0.001):
    model = Sequential(
        [
        Dense(128, activation='relu', input_dim=NUM_FEATURES),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  #softmax para multiclasse
        ]
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',  # ou 'categorical_crossentropy' para multiclasse
        metrics=['accuracy']
    )
    return model
