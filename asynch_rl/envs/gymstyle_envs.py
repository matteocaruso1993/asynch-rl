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
from .chess_env.chess_env import Chess
from .connect4_env.connect_four_env import ConnectFour

from .frx_env.frx_env import FrxTrdr


#
#
# conditional import of RobotEnv
"""
def modules_list() :
    #Return a list of available modules
    # Capture output of help into a string
    stdout_sys = sys.stdout
    stdout_capture = io.StringIO()
    sys.stdout = stdout_capture
    help('modules')
    sys.stdout = stdout_sys
    help_out = stdout_capture.getvalue()
    # Remove extra text from string
    help_out = help_out.replace('.', '')
    help_out = help_out.replace('available modules', '%').replace('Enter any module', '%').split('%')[-2]
    # Split multicolumn output
    help_out = help_out.replace('\n', '%').replace(' ', '%').split('%')
    help_out = list(filter(None, help_out))
    help_out.sort()
    return help_out

ml = modules_list()
"""
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
        if reset:
            state_tensor = (torch.from_numpy(state_obs[0]).unsqueeze(0).repeat(self.n_frames,1).unsqueeze(0).float(),\
                            torch.tensor(state_obs[1]).unsqueeze(0).float() )
        else:
            state_tensor = (torch.cat((state_tensor_z1[0].squeeze(0)[1:, :], torch.from_numpy(state_obs[0]).unsqueeze(0).float())).unsqueeze(0).float(), \
                            torch.tensor(state_obs[1]).unsqueeze(0).float() )
        return state_tensor
        
        
class DiscrGymStyleCartPole(DiscretizedActionWrapper):
    def __init__(self, n_bins_act=10, low_act=None, high_act=None, **kwargs):
        
        env = CartPoleEnv(**kwargs)
        super().__init__(env, n_bins_act, low_act, high_act)
        
class DiscrGymStylePlatoon(DiscretizedActionWrapper):
    def __init__(self, n_bins_act=10, low_act=None, high_act=None, **kwargs):
        
        env = PlatoonEnv(**kwargs)
        super().__init__(env, n_bins_act[:env.n_actions], low_act, high_act)
        
        
class GymstyleChess(DiscretizedActionWrapper):
    def __init__(self, n_bins_act=[15,7,7], **kwargs):
        
        env = Chess(**kwargs)
        super().__init__(env, n_bins_act)
        env.act_nD_flattened = self.act_nD_flattened
    

class GymstyleConnect4(DiscretizedActionWrapper):
    def __init__(self, **kwargs):
        
        env = ConnectFour(**kwargs)
        super().__init__(env, env.width-1)
        

class GymstyleDicePoker(DiscretizedActionWrapper):
    def __init__(self):
        
        env = DicePoker()
        super().__init__(env, 1)
        
        
class GymstyleFrx(DiscretizedActionWrapper):
    def __init__(self,  n_frames, max_n_moves,initial_account,low_act=None, high_act=None, **kwargs):
        
        env = FrxTrdr(n_samples = n_frames, max_n_moves = max_n_moves, initial_account = initial_account , **kwargs)
        super().__init__(env, [2, env.n_actions-1], low_act, high_act)
        
    def get_observations_structure(self):
        return None,None
    
    # get FRX specific NN input tensor
    def get_net_input(self, state_obs, **kwargs):
        #transforms env state output into compatible tensor with Network
        return ( torch.tensor(state_obs[0].T).unsqueeze(0).float(), torch.tensor(state_obs[1]).unsqueeze(0).float() )
    
       

        
#%%
if __name__ == "__main__":
    #env = RobotEnv()
    #my_env = GymStyle_Robot(n_bins_act=4, normalize_obs_state = False)
    my_env = DiscrGymStylePlatoon(n_bins_act=4)
