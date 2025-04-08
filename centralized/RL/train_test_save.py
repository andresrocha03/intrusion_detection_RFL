
from asyncio import sleep
import os
import time
import sys
import utils

sys.path.insert(0,'/home/andre/unicamp/ini_cien/intrusion_detection_RFL/environments')
import pandas as pd
import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN
from tabularenv import TabularEnv


def main():
    attack_type = "one" #one attack for binary classification, mul for multiclass classification

    #load dataset
    X_train, y_train = utils.load_train_data(attack_type)
    
    #choose model name
    selected = "DQN"
    option = f"{attack_type}_13"
    model_name = f"{selected}_{option}"

    #create model's and log's directories
    saving_model_dir = utils.create_model_dir(model_name)
    logdir = "logs/"

    #create training environment
    train_env = TabularEnv(X_train, y_train)
    
    #create model
    model = DQN("MlpPolicy", train_env, tensorboard_log=logdir, learning_rate=5e-5)
            ###### batch_size is how many transitions are selected from the replay buffer for eache gradient update
            ###### train_freq is how many steps are taken before each the Q network is updated
            ###### target_update_interval is how many steps are taken before each the target Q network is updated
            ###### buffer size is the replay buffer size


    #train model
    TIMESTEPS = 180000
    model, train_metrics = utils.train_dqn(TIMESTEPS, model, model_name, train_env)
  
    #save model
    model.save(f"{saving_model_dir}/{model_name}")   
    
    #load test data 
    X_test, y_test = utils.load_test_data(attack_type)

    #create test env
    test_env = TabularEnv(X_test, y_test)
    
    #load model
    #model_path = f"{saving_model_dir}/{model_name}.zip"
    #model = DQN.load(model_path,env=test_env)

    #test model
    test_metrics = utils.test_model(model, test_env, X_test, y_test, attack_type)    

    #print metrics
    print("----- TRAIN ------")
    print(train_metrics)
    print("----- TEST ------")
    print(test_metrics)

    #save test results
    utils.save_results(model_name, test_metrics, train_metrics['train_time'], attack_type)

main()
    

    