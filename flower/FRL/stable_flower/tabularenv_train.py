import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import numpy as np
import random
import pandas as pd


columns = 15
num_actions = 9

class TabularEnv(gym.Env):
    """
    Action Space:
    - Discrete space with two actions (0 or 1). For Classification 1 means benign and 0 means an attack

    Observation Space:
    - Box space with shape (1, _number of columns_) and dtype float32, representing a set of features for the intrusion data set.

    Methods:
    - step(action): Takes an action and returns the next observation, reward, done flag, and additional info.
    - reset(): Resets the environment to the initial state and returns the initial observation.
    - _next_obs(): Returns the next observation based on the current dataset and mode.

    Attributes:
    - action_space: Discrete space with two actions (0 or 1).
    - observation_space: Box space with shape (1, _number of  columns_) and dtype float32.
    - row_per_episode (int): Number of rows per episode.
    - step_count (int): Counter for the number of steps within the current episode.
    - x, y: Features and labels from the dataset.
    - random (bool): If True, observations are selected randomly from the dataset; otherwise, follows a sequential order.
    - dataset_idx (int): Index to keep track of the current observation in sequential mode.
    - expected_action (int): Expected action based on the current observation.
    """

    def __init__(self, dataset, row_per_episode=1 , random=False):
        super().__init__()

        # Define action space
        self.action_space = gym.spaces.Discrete(num_actions)

        # Define observation space
        observation = np.array([[np.finfo('float32').max] * columns], dtype=np.float32 )
        #observation = observation.flatten()
        self.observation_space = spaces.Box(-observation, observation, shape=(1,columns), dtype=np.float32)

        # Initialize parameters
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.correct = 0
        self.row_per_episode = row_per_episode
        self.step_count = 0
        self.x, self.y = dataset
        self.random = random
        self.current_obs = None
        self.dataset_idx = 0
        self.count = 0
        self.confusion_matrix = np.zeros(((num_actions), (num_actions)))

    def precision_recall(self, action):
        # print(f"expected: {self.expected_action} and got action: {action}")
        # print(f"matriz: {self.confusion_matrix[self.expected_action][action]}")
        self.confusion_matrix[self.expected_action][action] += 1
        # print(f"matriz: {self.confusion_matrix[self.expected_action][action]}")
        precision, recall = 0,0
        if (num_actions == 2):
            tp = self.confusion_matrix[1][1]
            fp = self.confusion_matrix[0][1]
            fn = self.confusion_matrix[1][0]
            # print(f"tp: {tp}   fp:{fp}   fn:{fn}")
            if ((tp + fp) == 0) or (tp+fn == 0):
                return precision, recall
            else:
                precision = float(tp) / float((tp + fp))
                recall = float(tp) / float((tp + fn))
                return precision, recall
        else: 
            precision_list = np.zeros((1,num_actions))
            recall_list    = np.zeros((1,num_actions))
            for i in range(num_actions):
                tp = self.confusion_matrix[i][i]
                fp = np.sum(self.confusion_matrix.T[i]) - tp
                fn = np.sum(self.confusion_matrix[i]) - tp
                if ((tp + fp) == 0) or (tp+fn == 0):
                    continue
                else:
                    precision_list[0][i] = float(tp) / float((tp + fp)) 
                    recall_list[0][i]    = float(tp) / float((tp + fn))
            precision = np.average(precision_list)
            recall = np.average(recall_list)
        return precision, recall
   
    def step(self, action):
        """
        Takes an action and returns the next observation, reward, done flag, and additional info.

        Parameters:
        - action (int): The action taken by the agent.

        Returns:
        - obs (numpy array): The next observation.
        - reward (int): The reward obtained based on the action.
        - done (bool): Flag indicating whether the episode is done.
        - info (dict): Additional information.
        """

        self.terminated = False
        if (int(action == self.expected_action)):
            reward = 1
            self.correct += 1
        else:
            reward = -1

        precision, recall = self.precision_recall(action) 

        self.count += 1

        obs = self._next_obs()
       

        self.step_count += 1
        if self.step_count >= self.row_per_episode:
            self.terminated = True
        
        info = {precision: precision, recall: recall}

        accuracy = float(self.correct)/float(self.count)
        print(f"step: {self.count}   index: {self.dataset_idx}")
        print(f"precision: {precision} and recall {recall} and accuracy {accuracy}")
        print("---------------------------------------------------")


        self.truncated = False
        return obs, reward, self.terminated, self.truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state and returns the initial observation.

        Returns:
        - obs (numpy array): The initial observation.
        """

        self.step_count = 0
        obs = self.x[self.dataset_idx]
        self.expected_action = int(self.y[self.dataset_idx])
        info = {}
        return obs, info

    def _next_obs(self):
        """
        Returns the next observation based on the current dataset and mode.

        Returns:
        - obs (numpy array): The next observation.
        """

        if self.random:
            next_obs_idx = random.randint(0, len(self.x) - 1)
            self.expected_action = int(self.y[next_obs_idx])
            obs = self.x[next_obs_idx]

        else:
            self.dataset_idx += 1
            if self.dataset_idx >= len(self.x):
                self.dataset_idx = 0
            obs = self.x[self.dataset_idx]
            self.expected_action = int(self.y[self.dataset_idx])
           

        return obs