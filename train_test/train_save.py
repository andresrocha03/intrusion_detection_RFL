
import os
import time
import sys

import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN


# sys.path.insert(0,'/home/andre/unicamp/IC/machine_learning/reinforcement_learning/tabular_data/environments')
from environments.tabularenv_train import TabularEnv



selected = "DQN"
option = "mul"
modelstr = f"{selected}_{option}"

models_dir = f"models/{modelstr}" 
logdir  = "logs/one_mul"


if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = TabularEnv()
env.reset()



TIMESTEPS = 140000
model1 = DQN("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
inicio = time.time()
model1.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=modelstr)
fim = time.time()

print(f'Tempo de treinamento: {fim-inicio:.2f}')
print(model1.get_parameters())
model2 = DQN("MlpPolicy", env)
print(model2.get_parameters())
model2.set_parameters(model1.get_parameters())
print(model2.get_parameters())


model1.save(f"{models_dir}/{selected}_{option}")   
