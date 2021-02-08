#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 08:57:26 2020

@author: Enrico Regolin
"""

# %%
import ray
import asyncio

from .updater import RL_Updater
from .sim_agent import SimulationAgent
from rl.utilities import progress



#%%
@ray.remote
class Ray_SimulationAgent(SimulationAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


#%%
@ray.remote(num_gpus=1)
class Ray_RL_Updater(RL_Updater):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    async def update_DQN_asynch(self):
        """asynchronous update of Reinforcement Learning Deep Network - only for Q-value"""
        
        self.sim_complete = stop_update = False
        total_loss = []
        epochs = 0
        
        if self.memory_pool is None:
            raise("memory pool is None object!!")
        
        print(f'Asynchronous update started: state_value')
        while not stop_update:

            loss = self.qValue_loss_update(*self.memory_pool.extractMinibatch()[:-1])
            total_loss.append(loss.cpu().item())
            epochs += 1

            await asyncio.sleep(0.00001)
            
            stop_update = self.sim_complete and (epochs >= self.n_epochs)
            
        print('')
        print(f'Training iteration completed. n_epochs: {epochs}')
        self.model_qv.model_version +=1
        
        return total_loss
        


    async def sentinel(self, sim_complete = True):
        """method coupled with update_DQN_asynch to periodically check if sim has finished"""
        if sim_complete:
            self.sim_complete = True
    
    async def get_current_QV_model(self):
        """method to extract current QV model version during update process (to speed up PG training)"""
        return self.model_qv
    
    async def update_model_v(self, model_v):
        """method to update model v"""
        self.model_v = model_v
