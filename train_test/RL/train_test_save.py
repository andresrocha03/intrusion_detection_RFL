
import os
import time
import sys
import utils

sys.path.insert(0,'/home/andre/unicamp/ini_cien/intrusion_detection_RFL/environments')
import pandas as pd
import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN
from tabularenv2 import TabularEnv


def main():
    #load dataset
    X_train, y_train = utils.load_train_data()
    
    #choose model name
    selected = "DQN"
    option = "one_4"
    model_name = f"{selected}_{option}"

    #create model and logs directory
    saving_model_dir = utils.create_model_dir(model_name)
    logdir = "logs/"

    #create training environment
    train_env = TabularEnv(X_train, y_train)
    train_env.reset()

    #create model
        # batch_size is how many transitions are selected from the replay buffer for eache gradient update
        # train_freq is how many steps are taken before each the Q network is updated
        # target_update_interval is how many steps are taken before each the target Q network is updated
        # buffer size is the replay buffer size
    model = DQN("MlpPolicy", train_env, tensorboard_log=logdir, batch_size=256)

    #train model
    TIMESTEPS = 100000
    model, train_time = utils.train_dqn(model, TIMESTEPS, model_name)
    print(f'Tempo de treinamento: {train_time:.2f}')

    #save model
    model.save(f"{saving_model_dir}/{model_name}")   

    #load test data 
    X_test, y_test = utils.load_test_data()

    #create test env
    test_env = TabularEnv(X_test, y_test)
    test_env.reset()

    #load model
    model_path = f"{saving_model_dir}/{model_name}.zip"
    model = DQN.load(model_path,env=test_env)

    #test model
    utils.test_model(model, test_env, X_test, y_test)    

main()
    

    