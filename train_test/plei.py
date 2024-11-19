import time
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from environments.tabularenv_train import TabularEnv


selected = "DQN"
model_path = f"/home/andre/unicamp/ini_cien/intrusion_detection_RFL/train_test/models/DQN_mul/DQN_mul.zip"

env = TabularEnv()
env.reset()

model_trained = DQN.load(model_path,env=env)
for key in model_trained.get_parameters()['policy.optimizer'].keys():
    print(key)

model_new  = DQN("MlpPolicy", env)
for key in model_new.get_parameters()['policy.optimizer'].keys():
    print(key)
