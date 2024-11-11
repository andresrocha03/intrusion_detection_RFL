import os
import pandas as pd
import time

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from tqdm import tqdm

from typing import List
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.callbacks import Callback

NUM_FEATURES = 15
NUM_UNIQUE_LABELS = 2


data_folder = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/processed_data/current_testing'


paths = ['x_one_train.csv', 'y_one_train.csv','x_one_test.csv','y_one_test.csv'] 

results_directory = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/plots'  # New directory for results
results_file = os.path.join(results_directory, 'results.csv')

results = pd.read_csv(results_file)

dfs_one = []
for path in paths:
    dfs_one.append(pd.read_csv(os.path.join(data_folder,path)))

paths = ['x_mul_train.csv','y_mul_train.csv','x_mul_test.csv','y_mul_test.csv']
dfs_mul = []
for path in paths:
    dfs_mul.append(pd.read_csv(os.path.join(data_folder,path)))

paths = ['x_sur_test.csv','y_sur_test.csv']
dfs_sur = []
for path in paths:
    dfs_sur.append(pd.read_csv(os.path.join(data_folder,path)))
      

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
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',  # ou 'categorical_crossentropy' para multiclasse
    metrics=['accuracy']
)


class TestLossCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_losses = []

    def on_epoch_end(self, epoch, logs=None):
        x_test, y_test = self.test_data
        loss, _ = self.model.evaluate(x_test, y_test, verbose=0)
        self.test_losses.append(loss)
        print(f'Test loss: {loss}')


test_callback = TestLossCallback((dfs_one[2].values, dfs_one[3].values))

# Train the model with validation data
history = model.fit(
    dfs_one[0],
    dfs_one[1],
    epochs=10,
    batch_size=32,
    validation_data=(dfs_one[2], dfs_one[3]),
    verbose=1
)

loss, accuracy = model.evaluate(dfs_one[2], dfs_one[3], verbose=0)

# Append model performance to the DataFrame
model_info = {
    'Model Name': 'tf-keras-central',
    'Loss': loss,
    'Accuracy': accuracy
}

new_row_df = pd.DataFrame([model_info])
results = pd.concat([results, new_row_df], ignore_index=True)
results.to_csv(results_file, index=False)

print(f"loss: {loss}, accuracy: {accuracy}")


# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

if not os.path.exists("plots"):
    os.makedirs("plots")

plt.savefig(os.path.join("plots", 'tf-keras-central.png'))
plt.show()