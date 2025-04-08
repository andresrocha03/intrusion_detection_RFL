from asyncio import sleep
import time
import os
import string
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from stable_baselines3 import A2C, PPO, DQN
import sys
sys.path.insert(0,'/home/andre/unicamp/ini_cien/intrusion_detection_RFL/environments')
from tabularenv import TabularEnv
import torch

def train_dqn(timesteps: int, model: DQN, model_name: string, train_env: TabularEnv) -> (DQN, dict):
    """
    Trains a DQN model for a given number of timesteps. 
    Saves the logs in a folder named with the model_name.
    Calculate and returns the metrics obtained from the training.
    
    Parameters:
    - timesteps (int): The number of timesteps to train the model.
    - model (DQN): The model to train.
    - model_name (string): The name of the model.
    - train_env (TabularEnv): The training environment.
    
    Returns:
    - model (DQN): The trained model.
    - metrics (dict): The metrics obtained from the training.
    """
    
    inicio = time.time()
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=model_name, progress_bar=True)
    fim = time.time()

    metrics = train_env.info
    metrics['train_time'] = fim-inicio
    
    return model, metrics

def load_train_data(attack_type: string):
    """
    Loads trains data from data directory

    Parameters:
    - attack_type (string): The type of attack to load the data from.
    
    Returns:
    - X_train (pd.DataFrame): The training features.
    - y_train (pd.DataFrame): The training labels.
    """
    # Load your dataset
    data_folder = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/processed_data/new_try/'
    X_train = pd.read_csv(os.path.join(data_folder, f"x_{attack_type}_train.csv" ))
    y_train = pd.read_csv(os.path.join(data_folder, f"y_{attack_type}_train.csv"))
    return X_train, y_train

def create_model_dir(model_name: string) -> string:
    """
    Creates a directory to save the model.

    Parameters:
    - model_name (string): The name of the model.

    Returns:
    - saving_model_dir (string): The directory to save the model.
    """
    saving_model_dir = f"models/{model_name}" 
    if not os.path.exists(saving_model_dir):
        os.makedirs(saving_model_dir)
    return saving_model_dir

def load_test_data(attack_type: string):
    """
    Loads the test data from the processed data folder.
    
    Parameters:
    - attack_type (string): The type of attack to load the data from.

    Returns:
    - X_test (pd.DataFrame): The test features.
    - y_test (pd.DataFrame): The test labels.
    """

    data_folder = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/processed_data/new_try/'
    X_test = pd.read_csv(os.path.join(data_folder, f"x_{attack_type}_test.csv" ))
    y_test = pd.read_csv(os.path.join(data_folder, f"y_{attack_type}_test.csv"))  

    return X_test, y_test

# Define the sigmoid function for binary classification
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the softmax function for multi-class
def softmax(x):
      # Convert PyTorch tensor to NumPy array if needed
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    if x.ndim == 1:
        x = x.reshape(1, -1)
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    
    return exps / np.sum(exps, axis=1, keepdims=True)

def predict_probabilities(model: DQN, obs: np.ndarray)-> np.ndarray:
    """
    Predicts the probabilities of each action for a given observation.
    
    Parameters:
    - model (DQN): The model to predict.
    - obs (np.ndarray): The observation to predict the probabilities for.

    Returns:
    - probabilities (np.ndarray): The probabilities of each action.
    """
    
    #extract q values
    probabilities = model.q_net.q_values_aux
    #calculate the probability of each action
    probabilities = softmax(probabilities)
    #reshape for 1D
    probabilities = probabilities.reshape(-1)

    return probabilities

def model_predict(model: DQN, obs: np.ndarray, attack_type: string, predictions:np.ndarray, probabilities:np.ndarray) -> int | np.ndarray:
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
    
    #predict the action
    action, _states = model.predict(obs)
    predictions.append(action)
    
    #predict the probabilities of each action if multiclass scenario
    if attack_type == "mul":
        prob_actions = predict_probabilities(model, obs) 
        probabilities.append(prob_actions)

    #return the action predicted
    return action

def test_model(model: DQN, test_env: TabularEnv, X_test: pd.DataFrame, y_test: pd.DataFrame, attack_type: string) -> dict:
    """
    Tests a model on a given test environment.

    Parameters:
    - model (DQN): The model to test.
    - test_env (TabularEnv): The test environment.
    - X_test (pd.DataFrame): The test features.
    - y_test (pd.DataFrame): The test labels.

    Returns:
    - metrics (dict): The metrics obtained from the testing.
    """

    # reset the environment to obtain initial observation
    obs, info = test_env.reset(seed=0)
   
    #predictions list, to be used in the metrics calculation
    predictions = []        

    #probabilities list, used only in the multiclass case
    probabilities = []
    
    #loop to predict the actions and append the predictions to the list
    inicio = time.time()     
    for _ in range(len(X_test)):
        action = model_predict(model, obs, attack_type, predictions, probabilities) 
        obs, rewards, terminated, truncated, info = test_env.step(action)
        if terminated:
            test_env.reset()        
    fim = time.time()

    #close the environment
    test_env.close()

    # calculate accuracy using the predictions
    classes_predictions = np.asarray(predictions)
    acc = accuracy_score(y_test, classes_predictions)

    #calculate the probabilities of the predictions
    if attack_type == "one":        
        prob_predictions = sigmoid(np.asarray(predictions))
    elif attack_type == "mul":
        prob_predictions = probabilities

    # print(prob_predictions)
    # calculate log loss
    loss = log_loss(y_test, prob_predictions)

    #create metrics dict
    metrics = {'accuracy': acc, 'loss': loss, 
               'precision': info['precision'], 'recall': info['recall'], 'test_time': fim-inicio}
    
    return metrics

def save_results(model_name, metrics, train_time, attack_type):
    
    model_info = {'Model Name': model_name, 
                    'Accuracy': metrics['accuracy'], 
                    'Loss': metrics['loss'], 
                    'Precision': metrics['precision'], 
                    'Recall': metrics['recall'], 
                    'Train Time': train_time, 
                 }
    
    results_directory = f'/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/results/central/{attack_type}' 
    results_file = os.path.join(results_directory, "dqn_res.csv")
    results = pd.read_csv(results_file)

    new_row = pd.DataFrame([model_info])
    results = pd.concat([results, new_row], ignore_index=True)
    results.to_csv(results_file, index=False)

def plot_confusion_matrix(confusion_matrix: np.ndarray, attack_type: string):
    """
    Plots the confusion matrix for a given attack type.

    Parameters:
    - confusion_matrix (np.ndarray): The confusion matrix to plot.
    - attack_type (string): The type of attack to plot the confusion matrix for.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    if attack_type=="one":
        df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "01"],
                  columns = [i for i in "01"])
    else:
        df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "012345678"],
                  columns = [i for i in "012345678"])
    
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.title(f'Confusion Matrix for DQN multiclass case')
    plt.show()
    #save the plot
    plt.savefig(f'cm_dqn_one.png')