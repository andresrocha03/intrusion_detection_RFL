import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_squared_error
from DQN_flower.tabularenv_train import TabularEnv

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        self.lr = 0.001
        self.gamma = 0.99
        self.exploration_proba = 1.0
        self.exploration_proba_decay = 0.00
        self.memory_buffer= list()
        self.model = Sequential([
            Dense(units=200,input_dim=state_size, activation = 'relu'),
            Dense(units=24,activation = 'relu'),
            Dense(units=action_size, activation = 'linear')
        ])
        self.model.compile(loss="mse",
                      optimizer = Adam(lr=self.lr))
    # the agent decides which action to perform
    def compute_action(self, current_state):
        if np.random.uniform(0,1) < self.exploration_proba:
            return np.random.choice(range(self.n_actions))
        q_values = self.model.predict(current_state)[0]
        return np.argmax(q_values)
    # we save each time step
    def store_episode(self,current_state, action, reward, next_state, done):
        self.memory_buffer.append({
            "current_state":current_state,
            "action":action,
            "reward":reward,
            "next_state":next_state,
            "done" :done
        })
    
    # when an episode is finished, we update the exploration probability
    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)
        print(self.exploration_proba)
    # train the model by replayin memory
    def train(self, batch_size):
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:batch_size]
        
        for experience in batch_sample:
            q_current_state = self.model.predict(experience["current_state"])
            q_target = experience["reward"]
            if not experience["done"]:
                q_target = q_target + self.gamma*np.max(self.model.predict(experience["next_state"])[0])
            q_current_state[0][experience["action"]] = q_target
            self.model.fit(experience["current_state"], q_current_state, verbose=0)


env = TabularEnv()
state_size = env.observation_space.shape[0]
action_size = 2
n_episodes = 1000
max_iteration_ep = 500
batch_size = 32



agent = DQNAgent(state_size, action_size)
total_steps = 0

for e in range(n_episodes):
    current_state = env.reset()
    current_state = np.array([current_state])
    rewards = 0
    for step in range(max_iteration_ep):
        total_steps = total_steps + 1
        action = agent.compute_action(current_state)
        
        next_state, reward, done, _ = env.step(action)
        next_state = np.array([next_state])
        rewards = rewards + reward
        agent.store_episode(current_state, action, reward, next_state, done)
        
        if done:
            agent.update_exploration_probability()
            break
        current_state = next_state
    print("episode ", e+1, " rewards: ", rewards)
    if total_steps >= batch_size:
        agent.train(batch_size=batch_size)
        
