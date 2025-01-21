import time
import os
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score


def train_dqn(timesteps, model, model_name):
    inicio = time.time()
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=model_name, progress_bar=True)
    fim = time.time()
    return model, fim-inicio

def load_train_data():
    # Load your dataset
    data_folder = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/processed_data/new_try/'
    df_train = pd.read_csv(os.path.join(data_folder, "x_one_train.csv" ))
    label_train = pd.read_csv(os.path.join(data_folder, "y_one_train.csv"))
    return df_train, label_train

def create_model_dir(model_name):
    saving_model_dir = f"models/{model_name}" 
    if not os.path.exists(saving_model_dir):
        os.makedirs(saving_model_dir)
    return saving_model_dir

def load_test_data():
    data_folder = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/processed_data/new_try/'
    X_test = pd.read_csv(os.path.join(data_folder, "x_one_test.csv" ))
    y_test = pd.read_csv(os.path.join(data_folder, "y_one_test.csv"))  
    return X_test, y_test

# Define the sigmoid function for binary classification
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the softmax function for multi-class
def softmax(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def test_model(model, env, X_test , y_test):
    attempts, correct = 0,0

    vec_env = model.get_env()
    obs = vec_env.reset()

    inicio = time.time()

    predictions = []
    for _ in range(22400):
        action, _states = model.predict(obs)
        predictions.append(action[0])
        obs, rewards, dones, info = vec_env.step(action)
        if rewards > 0:
            correct += 1
        print(_)
    fim = time.time()


    prob_predictions = sigmoid(np.asarray(predictions))
    acc = accuracy_score(y_test, np.asarray(predictions))
    loss = log_loss(y_test, prob_predictions)

    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {info[0]['precision']:.2f}")
    print(f"Recall: {info[0]['recall']:.2f}")
    print(f"tempo de teste: {fim-inicio:.2f}")

    env.close()
    return acc, loss, info[0]['precision'], info[0]['recall'], fim-inicio

def save_results(model_name, acc, loss, precision, recall, train_time):
    model_info = {'Model Name': model_name, 
                'Loss': loss, 'Accuracy': acc}

    results_directory = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/plots/central/one' 
    results_file = os.path.join(results_directory, "dqn_res.csv")
    results = pd.read_csv(results_file)

    new_row = pd.DataFrame([model_info])
    results = pd.concat([results, new_row], ignore_index=True)
    results.to_csv(results_file, index=False)

