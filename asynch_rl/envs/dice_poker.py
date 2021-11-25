#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:34:46 2021

@author: Enrico Regolin
"""

import numpy as np

import gym
from gym import spaces


#%%

class DicePoker(gym.Env):
    
    def __init__(self, reward = [-2, 5, 7, 8, 20] , threshold = 21):
        
        self.env_type = 'DicePoker'
        
        self.state = np.zeros(5)
        self.reward = reward
        self.threshold = threshold
        self.action_space = spaces.Box(low=np.zeros(5), high=np.ones(5)) 
        
    ###################################
    """ these three methods are needed in sim env / rl env"""
    def get_max_iterations(self):
        return 2
 
    def get_actions_structure(self):
        pass # implemented in wrapper

    def get_observations_structure(self):
        return 5
    ###################################
        
    def reset(self, **kwargs):
        self.state = np.sort(np.random.randint(1,7,5))
        self.n_rethrows = 0
        
        return self.state/6
        
    def step(self, action, *args ):
        
        info = {'outcome': None}
        done = False
        
        if any(action):
            rethrow = np.random.randint(1,7,np.sum(action.astype(bool)))
            self.state[action.astype(bool)] = rethrow
            self.state = np.sort(self.state)
            reward = -np.sum(action.astype(bool))
            self.n_rethrows += 1
        else:
            reward = 0
            done = True
            info['result'] = 0
            info['no step'] = None
        
        success = self.success_condition() 
        
        #if not success and self.n_rethrows == 1:
        #    stophere=1
        
        if success or self.n_rethrows >= 2:
            done = True
            reward += self.reward[success]
            info['result'] = success
            #if success:
            #    print(self.state)
        
        return self.state/6, reward, done, info


    def success_condition(self):
        if np.sum(self.state) >= self.threshold:
            if np.max(np.bincount(self.state)[1:]) >= 3:
                return 1
        if np.equal(np.sort(self.state) , np.array([1,2,3,4,5]) ).all() or np.equal(np.sort(self.state) , np.array([2,3,4,5,6]) ).all():
            return 2
        if any([np.sum(self.state == i)==4 for i in range(1,7)]) :
            return 3
        if any([np.all(self.state == i) for i in range(1,7)]) :
            return 4
        return 0

    