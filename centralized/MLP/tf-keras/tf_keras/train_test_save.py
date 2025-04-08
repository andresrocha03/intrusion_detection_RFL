import os
import pandas as pd
import time
import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import keras
from keras.callbacks import Callback
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from typing import List
from numpy.typing import NDArray
import tensorflow as tf


NUM_FEATURES = 15
NUM_UNIQUE_LABELS = 9 # 2 for binary classification, 9 for multiclass classification
attack_type = 'mul'  # 'one' or 'multiple'


# Load the results file
results_directory = f'/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/plots/central/{attack_type}'  # New directory for results
results_file = os.path.join(results_directory, 'mlp_results.csv')
results = pd.read_csv(results_file)

# Load the data
data_folder = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/processed_data/new_try'
paths = [f'x_{attack_type}_train.csv', f'y_{attack_type}_train.csv',f'x_{attack_type}_test.csv',f'y_{attack_type}_test.csv'] 
data = []
for path in paths:
    data.append(pd.read_csv(os.path.join(data_folder,path)))
  
# Guarantee the right parameters for each data
if attack_type == 'one':
    activation = 'sigmoid'
    loss = 'crossentropy'
    average = 'binary'

elif attack_type == 'mul':
    activation = 'softmax'
    loss = 'sparse_categorical_crossentropy'
    average = 'macro'


# Create model 
model = Sequential(
[
    Dense(NUM_FEATURES, activation='relu', input_dim=NUM_FEATURES),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(NUM_UNIQUE_LABELS, activation=activation)  # sigmoid for binary classification
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss=loss,  
    metrics=['accuracy']
)

# create evaluate function
class TestLossCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_losses = []

    def on_epoch_end(self, epoch, logs=None):
        x_test, y_test = self.test_data
        loss, _ = self.model.evaluate(x_test, y_test, verbose=0)
        self.test_losses.append(loss)
        print(f'Test loss: {loss}')

# Train the model_one with training data
history = model.fit(
    data[0],
    data[1],
    epochs=10,
    batch_size=64,
    validation_data=(data[2], data[3]),
    verbose=1
)

loss, accuracy = model.evaluate(data[2], data[3], verbose=0)

# get precision and recall of the model
y_pred = model.predict(data[2])
y_pred = np.argmax(y_pred, axis=1)
precision = precision_score(data[3], y_pred, average=average, zero_division=0)
recall = recall_score(data[3], y_pred, average=average, zero_division=0)

# plot the confusion matrix
cm = confusion_matrix(data[3], y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(data[3]))
disp.plot()
plt.savefig('tf-keras-central-cm.png')

test_callback = TestLossCallback((data[2].values, data[3].values))

# Append model performance to the DataFrame
model_info = {
    'Model Name': 'MLP',
    'Loss': loss,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'Train Time': history.epoch[-1],
}

new_row_df = pd.DataFrame([model_info])
results = pd.concat([results, new_row_df], ignore_index=True)
results.to_csv(results_file, index=False)

print(f"loss: {loss}, accuracy: {accuracy}")

# Save model
model.save(f'/home/andre/unicamp/ini_cien/intrusion_detection_RFL/train_test/MLP/tf-keras/tf_keras/models/{attack_type}_model.keras')

# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('mlp-central.png')
plt.show()
