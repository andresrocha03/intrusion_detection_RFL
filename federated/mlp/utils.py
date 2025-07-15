import os

from typing import List
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy, Precision, Recall

from sklearn.model_selection import train_test_split

NUM_UNIQUE_LABELS = 2  # Number of unique labels in your dataset
NUM_FEATURES = 15  # Number of features in your dataset


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
    X_train = pd.read_csv(os.path.join(data_folder, "x_one_train.csv" ))
    y_train = pd.read_csv(os.path.join(data_folder, "y_one_train.csv"))
    X_train['label'] = y_train
    df_train = X_train

    X_test = pd.read_csv(os.path.join(data_folder, "x_one_test.csv"))
    y_test = pd.read_csv(os.path.join(data_folder, "y_one_test.csv"))
    X_test['label'] = y_test
    df_test = X_test

    return df_train, df_test


def load_surprise_dataset(data_folder: str) -> tuple[NDArray, NDArray, NDArray, NDArray]:  
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

def load_data(partition: list[NDArray]):
    """Load data."""
    X = partition.drop('label', axis=1).values
    y = partition['label'].values
    return X, y 

def partition_data(data, num_partitions):
# Partitioning the dataset into parts for each client
    return np.array_split(data, num_partitions)

def load_model(learning_rate=0.000005):
    model = Sequential(
        [
        Dense(NUM_FEATURES, activation='relu', input_dim=NUM_FEATURES),
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


def get_scores(y_true, y_pred):
    """
    Get scores.

    Parameters:
    - y_true: np.ndarray
        True labels.
    - y_pred: np.ndarray
        Predicted labels.
    
    Returns:
    - scores: dict
        Dictionary containing the scores.
    """
    
    scores = {}

    y_true = y_true.reshape(-1, 1) #reshape to 2D array 
    
    # Calculate mean, the function returns the loss per batch. 
    # Convert to float for flower to accept the value
    scores['loss'] = float((tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred)).numpy()))

    y_pred = y_pred.reshape(-1, 1) #reshape to 2D array
    
    scores['accuracy'] = float(tf.reduce_mean(keras.metrics.binary_accuracy(y_true, y_pred)).numpy())
   
    # For precision and recall, we need to work with 1D arrays of labels.
    # Depending on your model output (probabilities vs. logits), you might need:
    # For binary classification: threshold the predictions.
    # For multi-class classification: use np.argmax.
    #
    # Here we assume binary classification. Adjust as needed.
    y_pred = (y_pred > 0.5).astype(int).reshape(-1)
    y_true_labels = y_true.reshape(-1)

    # Calculate Precision
    precision_metric = keras.metrics.Precision()
    precision_metric.update_state(y_true_labels, y_pred)
    scores['precision'] = float(precision_metric.result().numpy())

    # Calculate Recall
    recall_metric = keras.metrics.Recall()
    recall_metric.update_state(y_true_labels, y_pred)
    scores['recall'] = float(recall_metric.result().numpy())
    
    return scores
