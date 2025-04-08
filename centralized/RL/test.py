import utils
import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN
from tabularenv import TabularEnv

attack_type = "mul" #one attack for binary classification, mul for multiclass classification

#load test data 
X_test, y_test = utils.load_test_data(attack_type)

#create test env
test_env = TabularEnv(X_test, y_test)

#load model
saving_model_dir = "models/"
model_name = f"DQN_{attack_type}_1"

model_path = f"{saving_model_dir}/{model_name}/{model_name}"
model = DQN.load(model_path,env=test_env)

#test model
test_metrics = utils.test_model(model, test_env, X_test, y_test, attack_type)   

confusion_matrix = test_env.confusion_matrix
#plot confusion matrix
utils.plot_confusion_matrix(confusion_matrix, attack_type)

#print metrics
print("----- TEST ------")
print(test_metrics)