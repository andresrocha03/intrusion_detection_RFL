import sys
import os
# adding environment to the system path
sys.path.insert(0, '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/environments')
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from tabularenv import TabularEnv
import pandas as pd
from tabularenv import TabularEnv


selected = "DQN"
model_path = f"/home/andre/unicamp/ini_cien/intrusion_detection_RFL/train_test/models/DQN_mul/DQN_mul.zip"

# Load your dataset
data_folder = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/processed_data/current_testing/'
X_train = pd.read_csv(os.path.join(data_folder, "x_one_train.csv" ))
y_train = pd.read_csv(os.path.join(data_folder, "y_one_train.csv"))
X_test = pd.read_csv(os.path.join(data_folder, "x_one_test.csv"))
y_test = pd.read_csv(os.path.join(data_folder, "y_one_test.csv"))

env = TabularEnv(X_train, y_train)
env.reset()
env_test = TabularEnv(X_test, y_test)
env.reset()


model_train  = DQN("MlpPolicy", env)
for i in range(5):
    model_train.learn(total_timesteps=10000, reset_num_timesteps=False, progress_bar=True)
    model_test = DQN("MlpPolicy", env_test)
    model_test.set_parameters(model_train.get_parameters())

    terminated = False
    predictions = []
    vec_env = model_test.get_env()
    obs = vec_env.reset(seed=0)
    while not terminated:
        action, _states = model_test.predict(obs)
        predictions.append(action[0])
        obs, rewards, dones, info = vec_env.step(action)
        terminated = info[0]["terminated"]
    print(predictions)
    accuracy = predictions.count(1)/len(predictions)
    print(f"Accuracy: {accuracy:.2f}")

    mean_reward, std_reward = evaluate_policy(model_test, model_test.get_env(), n_eval_episodes=5000)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")