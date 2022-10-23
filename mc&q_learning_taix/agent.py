import numpy as np
import random

from collections import defaultdict
class Agent:

    def __init__(self, Q, mode="mc_contral"):
        self.Q = Q
        self.mode = mode
        self.n_actions = 6
        self.gamma = 0.85
        if mode == "mc_control":
            self.step = self.step_mc
            self.learning_rate = 0.01
            self.e = list()
        else:
            self.step = self.step_ql
            self.learning_rate = 0.2           # Learning rate


    
    def select_action(self, state, eps = 0):
        # eps  = 1.0 / ((i_episode // 100) + 1) during training
        exp_tradeoff = random.random()

        ## If this number > greater than epsilon --> exploitation 
        if exp_tradeoff > eps:
            action = np.argmax(self.Q[state])

        # Else doing a random choice --> exploration
        else:
            action =  np.random.choice(self.n_actions)


        return action
    def step_mc(self, state, action, reward, next_state, done):
        
        if done:
            rewards = defaultdict(lambda: np.zeros(self.n_actions))
            for his in reversed(self.e):
                state, action, reward = his
                rewards[state][action] = reward + self.gamma * rewards[state][action]
                self.Q[state][action] += self.learning_rate * (rewards[state][action] - self.Q[state][action])
            self.e.clear()
        else:
            self.e.append((state, action, reward))
    def step_ql(self, state, action, reward, next_state, done):
       
        self.Q[state][action] = self.Q[state][action] + self.learning_rate * (reward + self.gamma *np.max(self.Q[next_state]) - self.Q[state][action])
     
            
 
        

          

        