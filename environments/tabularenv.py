import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import numpy as np
import random
import pandas as pd

columns = 15
num_actions = 2

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

    def __init__(self,  df_train_x, df_train_y, row_per_episode=100, random=False):
        super().__init__()

        # Define action space
        self.action_space = gym.spaces.Discrete(num_actions)

        # Define observation space
        observation = np.array([[np.finfo('float32').max] * columns], dtype=np.float32 )
        #observation = observation.flatten()
        self.observation_space = spaces.Box(-observation, observation, shape=(1,columns), dtype=np.float32)

        df_train_x = np.expand_dims(df_train_x, 1)
        df_train_y = np.expand_dims(df_train_y, 1)

        # Initialize parameters
        self.confusion_matrix = np.zeros((num_actions, num_actions))
        self.row_per_episode = row_per_episode
        self.step_count = 0
        self.x, self.y = df_train_x, df_train_y
        self.random = random
        self.current_obs = None
        self.dataset_idx = 0
        self.acc_count = 0
        self.total_rwd = 0
        self.terminated = False
        self.info = {}

    def precision_recall(self, action):
            # update confusion matrix
            self.confusion_matrix[self.expected_action][action] += 1

            # initialize precision and recall
            precision, recall = 0,0

            # check for binary or multi-class classification
            if (num_actions == 2): #binary classification
                tp = self.confusion_matrix[1][1] #true-positives
                fp = self.confusion_matrix[0][1] #false-positives
                fn = self.confusion_matrix[1][0] #false-negatives
                
                if ((tp + fp) == 0) or (tp+fn == 0): #check for division by zero
                    return precision, recall
                else: #calculate precision and recall
                    precision = float(tp) / float((tp + fp))
                    recall = float(tp) / float((tp + fn))
                    return precision, recall
            
            else: #multi-class classification
                # initialize precision list for each class
                precision_list = np.zeros((1,num_actions)) 
                recall_list    = np.zeros((1,num_actions))

                for i in range(num_actions):
                    tp = self.confusion_matrix[i][i]
                    fp = np.sum(self.confusion_matrix.T[i]) - tp
                    fn = np.sum(self.confusion_matrix[i]) - tp
                    if ((tp + fp) == 0) or (tp+fn == 0): #check for division by zero
                        continue
                    else:  
                        precision_list[0][i] = float(tp) / float((tp + fp)) #calculate precision for class i 
                        recall_list[0][i]    = float(tp) / float((tp + fn)) #calculate recall for class i
                #use average precision and recall
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
        - terminated (bool): Flag indicating whether the episode is done.
        - info (dict): Additional information.
        """

        self.step_count += 1
        self.truncated = False
        self.terminated = False

        reward = 0
        if (int(action) == 0 and self.expected_action==0):
            reward = 3
            self.total_rwd += 1
        elif (int(action) == 1 and self.expected_action==1):
            reward = 10
            self.total_rwd += 1
        elif (int(action)==0 and self.expected_action==1):
            reward = -10
        elif (int(action)==1 and self.expected_action==0):
            reward = -3


        precision, recall = self.precision_recall(action)
        # precision = 0
        # recall = 0

        self.acc_count += 1
        accuracy = self.total_rwd / self.acc_count
      
        self.info = {"step": self.acc_count, "idx": self.dataset_idx, "accuracy": accuracy, "precision": precision, "recall": recall, "terminated": self.terminated}
        # print(self.info)

      
        obs = self.x[self.dataset_idx]
        if self.step_count >= self.row_per_episode:
            self.terminated = True
        else:
            obs = self._next_obs()
        
        return obs, reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None, options=None):
        """
        Resets the environment to the initial state and returns the initial observation.

        Returns:
        - obs (numpy array): The initial observation.
        """

        # print(f"reset called, reached instance count {self.acc_count}")
        self.step_count = 0
    
        if seed==0:
            obs = self.x[seed]
            self.expected_action = int(self.y[seed])
            return obs, self.info
        
        obs = self._next_obs()
        return obs, self.info

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
                # self.terminated = True
            
            obs = self.x[self.dataset_idx]
            self.expected_action = int(self.y[self.dataset_idx])

        return obs