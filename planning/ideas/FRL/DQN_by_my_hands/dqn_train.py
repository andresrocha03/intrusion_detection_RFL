import os
import time
import sys

import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN
import pandas as pd

sys.path.insert(0,'/home/andre/unicamp/ini_cien/intrusion_detection_RFL/enviroments')
from environments.tabularenv_train import TabularEnv
from dqn_agent import DQN

# Load your dataset
data_folder = '/home/andre/unicamp/ini_cien/intrusion_detection_RFL/data/processed_data/current_testing/'
df_train = pd.read_csv(os.path.join(data_folder, "x_one_train.csv" ))
label_train = pd.read_csv(os.path.join(data_folder, "y_one_train.csv"))
df_train['label'] = label_train
df_test = pd.read_csv(os.path.join(data_folder, "x_one_test.csv"))
label_test = pd.read_csv(os.path.join(data_folder, "y_one_test.csv"))
df_test['label'] = label_test


#create our environment
env = TabularEnv()
env.reset()

action_size = env.action_space.n
agent = DQN(action_size)

episodes = 500
batch_size = 8
skip_start = 90  # MsPacman-v0 waits for 90 actions before the episode begins
total_time = 0   # Counter for total number of steps taken
all_rewards = 0  # Used to compute avg reward over time
blend = 4        # Number of images to blend
done = False

for e in range(episodes):
    total_reward = 0
    game_score = 0
    state = process_frame(env.reset())
    images = deque(maxlen=blend)  # Array of images to be blended
    images.append(state)
    
    for skip in range(skip_start): # skip the start of each game
        env.step(0)
    
    for time in range(20000):
        env.render()
        total_time += 1
        
        # Every update_rate timesteps we update the target network parameters
        if total_time % agent.update_rate == 0:
            agent.update_target_model()
        
        # Return the avg of the last 4 frames
        state = blend_images(images, blend)
        
        # Transition Dynamics
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        # Return the avg of the last 4 frames
        next_state = process_frame(next_state)
        images.append(next_state)
        next_state = blend_images(images, blend)
        
        # Store sequence in replay memory
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        game_score += reward
        reward -= 1  # Punish behavior which does not accumulate reward
        total_reward += reward
        
        if done:
            all_rewards += game_score
            
            print("episode: {}/{}, game score: {}, reward: {}, avg reward: {}, time: {}, total time: {}"
                  .format(e+1, episodes, game_score, total_reward, all_rewards/(e+1), time, total_time))
            
            break
            
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

agent.save('models/5k-memory_1k-games')