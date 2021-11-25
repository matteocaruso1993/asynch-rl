# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:24:29 2020

@author: Enrico Regolin
"""

#%%
import os
import torch

import sys
import io


from .wrappers import DiscretizedActionWrapper,DiscretizedObservationWrapper, ContinuousHybridActionWrapper

from .dice_poker import DicePoker

from .cartpole_env.cartpole_env import CartPoleEnv
from .platoon_env.platooning_env import PlatoonEnv


try:
    import robot_sf
    robot_available = True
except Exception:
    robot_available = False

if robot_available:
    from .robot_env.robot_env import RobotEnv
else:
    from .robot_env.robot_env_dummy import RobotEnv

#%%

#env = RobotEnv(self.normalize_obs_state = True)

class DiscrGymStyleRobot(DiscretizedActionWrapper):
    def __init__(self, n_frames, n_bins_act=10, low_act=None, high_act=None, **kwargs):
        self.n_frames = n_frames
        env = RobotEnv(**kwargs)
        super().__init__(env, n_bins_act, low_act, high_act)
        
    # get robot specific NN input tensor
    def get_net_input(self, state_obs, state_tensor_z1 = None, reset = False):
        """ transforms env state output into compatible tensor with Network"""
        
        new_full_state_observation = torch.cat((torch.from_numpy(state_obs[0]).unsqueeze(0),torch.from_numpy(state_obs[1][:2]).unsqueeze(1).repeat(1,state_obs[0].shape[0]))).float()
        new_robot_state = torch.tensor(state_obs[1]).unsqueeze(0).float()
        
        if reset or state_tensor_z1 is None:
            state_tensor = (new_full_state_observation.unsqueeze(1).repeat(1,self.n_frames,1).unsqueeze(0), new_robot_state )
        else:
            state_tensor = (torch.cat((state_tensor_z1[0].squeeze(0)[:,1:, :], new_full_state_observation.unsqueeze(1)),dim = 1).unsqueeze(0), new_robot_state )
            
        return state_tensor
    
    """
    # get robot specific NN input tensor
    def get_net_input(self, state_obs, state_tensor_z1 = None, reset = False):
        #transforms env state output into compatible tensor with Network
        if reset:
            state_tensor = (torch.from_numpy(state_obs[0]).unsqueeze(0).repeat(self.n_frames,1).unsqueeze(0).float(),\
                            torch.tensor(state_obs[1]).unsqueeze(0).float() )
        else:
            state_tensor = (torch.cat((state_tensor_z1[0].squeeze(0)[1:, :], torch.from_numpy(state_obs[0]).unsqueeze(0).float())).unsqueeze(0).float(), \
                            torch.tensor(state_obs[1]).unsqueeze(0).float() )
        return state_tensor
    """

        
        
class DiscrGymStyleCartPole(DiscretizedActionWrapper):
    def __init__(self, n_bins_act=10, low_act=None, high_act=None, **kwargs):
        
        env = CartPoleEnv(**kwargs)
        super().__init__(env, n_bins_act, low_act, high_act)
        
class DiscrGymStylePlatoon(DiscretizedActionWrapper):
    def __init__(self, n_bins_act=10, low_act=None, high_act=None, **kwargs):
        
        env = PlatoonEnv(**kwargs)
        super().__init__(env, n_bins_act[:env.n_actions], low_act, high_act)
        

class GymstyleDicePoker(DiscretizedActionWrapper):
    def __init__(self):
        
        env = DicePoker()
        super().__init__(env, 1)
        

        
#%%
if __name__ == "__main__":
    #env = RobotEnv()
    #my_env = GymStyle_Robot(n_bins_act=4, normalize_obs_state = False)
    my_env = DiscrGymStylePlatoon(n_bins_act=4)
