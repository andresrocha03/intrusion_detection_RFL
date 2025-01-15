import time
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from train_test.environments.tabularenv import TabularEnv
from sklearn.metrics import log_loss, accuracy_score

import os
import pandas as pd
import numpy as np

selected = "DQN"
option = "mul"
model_path = f"models/{option}_attack/{selected}_{option}/{selected}_{option}.zip"

env = TabularEnv()
env.reset()

model = DQN.load(model_path,env=env)

#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)

attempts, correct = 0,0

vec_env = model.get_env()
obs = vec_env.reset()
inicio = time.time()
terminated = False

# while (not terminated):
predictions = []
for _ in range(15026):
    action, _states = model.predict(obs)
    predictions.append(action[0])
    obs, rewards, dones, info = vec_env.step(action)
    attempts += 1
    if rewards > 0:
        correct += 1
    terminated = info[0]["terminated"]

fim = time.time()
accuracy = correct/attempts
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {info[0]['precision']:.2f}")
print(f"Recall: {info[0]['recall']:.2f}")
print(f"tempo de teste: {fim-inicio:.2f}")
env.close()

data_folder = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/processed_data/current_testing/'
path_y = 'y_mul_test.csv'
test_y = pd.read_csv(os.path.join(data_folder,path_y))

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the softmax function
def softmax(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

prob_predictions = softmax(np.asarray(predictions))

loss = log_loss(test_y, prob_predictions)
print(f"loss {loss}")
acc = accuracy_score(test_y, np.asarray(predictions))
print(f"acc: {acc}")
model_info = {'Model Name': 'DQN', 
              'Loss': loss, 'Accuracy': acc}

results_directory = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/plots/central/mul' 
results_file = os.path.join(results_directory, "dqn_res.csv")
results = pd.read_csv(results_file)

new_row = pd.DataFrame([model_info])
results = pd.concat([results, new_row], ignore_index=True)
results.to_csv(results_file, index=False)
