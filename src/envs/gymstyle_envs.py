# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:24:29 2020

@author: Enrico Regolin
"""

#%%
import os
import torch

from envs.wrappers import DiscretizedActionWrapper,DiscretizedObservationWrapper, ContinuousHybridActionWrapper

from envs.DicePoker import DicePoker

from envs.cartpole_env.CartPole_env import CartPoleEnv
from envs.platoon_env.Platooning_env import PlatoonEnv
from envs.chess_env.Chess_env import Chess
from envs.connect4_env.ConnectFour_env import ConnectFour

from envs.frx_env.frx_env import FrxTrdr
# conditional import of RobotEnv

if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.join('robot_env','src')) ):
    from envs.robot_env.Robot_env import RobotEnv
else:
    from envs.robot_env.Robot_env_dummy import RobotEnv



#env = RobotEnv(self.normalize_obs_state = True)

class discr_GymStyle_Robot(DiscretizedActionWrapper):
    def __init__(self, n_frames, n_bins_act=10, low_act=None, high_act=None, **kwargs):
        self.n_frames = n_frames
        env = RobotEnv(**kwargs)
        super().__init__(env, n_bins_act, low_act, high_act)
        
    # get robot specific NN input tensor
    def get_net_input(self, state_obs, state_tensor_z1 = None, reset = False):
        """ transforms env state output into compatible tensor with Network"""
        if reset:
            state_tensor = torch.from_numpy(state_obs).unsqueeze(0).repeat(self.n_frames,1).unsqueeze(0).float()
        else:
            state_tensor = torch.cat((state_tensor_z1.squeeze(0)[1:, :], torch.from_numpy(state_obs).unsqueeze(0))).unsqueeze(0).float()
        return state_tensor
        
        
        
class discr_GymStyle_CartPole(DiscretizedActionWrapper):
    def __init__(self, n_bins_act=10, low_act=None, high_act=None, **kwargs):
        
        env = CartPoleEnv(**kwargs)
        super().__init__(env, n_bins_act, low_act, high_act)
        
class discr_GymStyle_Platoon(DiscretizedActionWrapper):
    def __init__(self, n_bins_act=10, low_act=None, high_act=None, **kwargs):
        
        env = PlatoonEnv(**kwargs)
        super().__init__(env, n_bins_act[:env.n_actions], low_act, high_act)
        
class contn_hyb_GymStyle_Platoon(ContinuousHybridActionWrapper):
    def __init__(self,action_structure, low_act=None, high_act=None, **kwargs):
        
        env = PlatoonEnv(**kwargs)
        super().__init__(env, action_structure[:env.n_actions], low_act, high_act)
        
class Gymstyle_Chess(DiscretizedActionWrapper):
    def __init__(self, n_bins_act=[15,7,7], **kwargs):
        
        env = Chess(**kwargs)
        super().__init__(env, n_bins_act)
        env.act_nD_flattened = self.act_nD_flattened
    

class Gymstyle_Connect4(DiscretizedActionWrapper):
    def __init__(self, **kwargs):
        
        env = ConnectFour(**kwargs)
        super().__init__(env, env.width-1)
        

class Gymstyle_DicePoker(DiscretizedActionWrapper):
    def __init__(self):
        
        env = DicePoker()
        super().__init__(env, 1)
        
class Gymstyle_Frx():
    def __init__(self, n_frames, max_n_moves,initial_account, **kwargs):
        
        self.env = FrxTrdr(n_samples = n_frames, max_n_moves = max_n_moves, initial_account = initial_account , **kwargs)
        self.n_bins_act = self.env.n_actions - 1
        self.env_type = self.env.env_type
        
    def get_actions_structure(self):
        return self.n_bins_act+1

    def get_observations_structure(self):
        return None,None
    
    # get FRX specific NN input tensor
    def get_net_input(self, state_obs, **kwargs):
        """ transforms env state output into compatible tensor with Network"""
        return ( torch.tensor(state_obs[0].T).unsqueeze(0).float(), torch.tensor(state_obs[1]).unsqueeze(0).float() )
    
    def action(self, action_bool_array):
        return self.env.step(action_bool_array)

    def reset(self, **kwargs):
        return self.env.reset()
    
    def get_max_iterations(self):
        return self.env.max_n_moves



        
#%%
if __name__ == "__main__":
    #env = RobotEnv()
    #my_env = GymStyle_Robot(n_bins_act=4, normalize_obs_state = False)
    my_env = discr_GymStyle_Platoon(n_bins_act=4)
