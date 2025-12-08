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
from sklearn.metrics import precision_score, recall_score

NUM_UNIQUE_LABELS = 9  # Number of unique labels in your dataset
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
    X_train = pd.read_csv(os.path.join(data_folder, "x_mul_train.csv" ))
    y_train = pd.read_csv(os.path.join(data_folder, "y_mul_train.csv"))
    X_train['label'] = y_train
    df_train = X_train

    X_test = pd.read_csv(os.path.join(data_folder, "x_mul_test.csv"))
    y_test = pd.read_csv(os.path.join(data_folder, "y_mul_test.csv"))
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
        Dense(NUM_UNIQUE_LABELS, activation='softmax')  #sigmoid ou softmax
        ]
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',  # binary_crossentropy ou 'categorical_crossentropy' 
        metrics=['accuracy']
    )
    return model


def get_binary_scores(y_true, y_pred):
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

    #binary classification
    # y_true = y_true.reshape(-1, 1) #reshape to 2D array 
    
    # Calculate mean, the function returns the loss per batch. 
    # Convert to float for flower to accept the value
    
    print(y_true.shape, y_pred.shape)
    print(y_pred)
    ## scores['loss'] = float((tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred)).numpy()))
    scores['loss'] = float((tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y_true, y_pred)).numpy()))

    # binary classification
    ## y_pred = y_pred.reshape(-1, 1) #reshape to 2D array

    # scores['accuracy'] = float(tf.reduce_mean(keras.metrics.binary_accuracy(y_true, y_pred)).numpy())
    scores['accuracy'] = float(tf.reduce_mean(keras.metrics.sparse_categorical_accuracy(y_true, y_pred)).numpy())

    # For precision and recall, we need to work with 1D arrays of labels.
    # Depending on your model output (probabilities vs. logits), you might need:
    # For binary classification: threshold the predictions.
    # For multi-class classification: use np.argmax.
   
    # Here we assume binary classification. Adjust as needed.
    # y_pred = (y_pred > 0.5).astype(int).reshape(-1)
    y_pred = np.argmax(y_pred, axis=1).reshape(-1)
    y_true_labels = y_true.reshape(-1)

    # Calculate Precision 
    precision_metric = keras.metrics.Precision()
    precision_metric.update_state(y_true_labels, y_pred, average='macro', zero_division=0)
    scores['precision'] = float(precision_metric.result().numpy())

    # Calculate Recall
    recall_metric = keras.metrics.Recall()
    recall_metric.update_state(y_true_labels, y_pred, average='macro', zero_division=0)
    scores['recall'] = float(recall_metric.result().numpy())
    
    return scores

def get_multiclasses_scores(y_true, y_pred_probs):
    """
    y_true: (N,) integer class labels
    y_pred_probs: (N, NUM_UNIQUE_LABELS) softmax probabilities
    """
   
    y_true = np.asarray(y_true).reshape(-1)
    y_pred_probs = np.asarray(y_pred_probs)

    # ----- Loss & Accuracy (sparse) -----
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred_probs)
    loss = float(tf.reduce_mean(loss).numpy())

    acc = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred_probs)
    acc = float(tf.reduce_mean(acc).numpy())

    # ----- Convert probabilities -> predicted class indices -----
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    # ----- Multi-class precision/recall -----
    precision = float(precision_score(y_true, y_pred_labels, average="macro", zero_division=0))
    recall    = float(recall_score(y_true,  y_pred_labels, average="macro", zero_division=0))

    return {"loss": loss, "accuracy": acc, "precision": precision, "recall": recall}
