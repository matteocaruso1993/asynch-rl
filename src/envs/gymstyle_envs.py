# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:24:29 2020

@author: Enrico Regolin
"""

#%%
import os

from envs.wrappers import DiscretizedActionWrapper,DiscretizedObservationWrapper, ContinuousHybridActionWrapper


from envs.cartpole_env.CartPole_env import CartPoleEnv
from envs.platoon_env.Platooning_env import PlatoonEnv
from envs.chess_env.Chess_env import Chess
from envs.connect4_env.ConnectFour_env import ConnectFour
# conditional import of RobotEnv

if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.join('robot_env','src')) ):
    from envs.robot_env.Robot_env import RobotEnv
else:
    from envs.robot_env.Robot_env_dummy import RobotEnv



#env = RobotEnv(self.normalize_obs_state = True)

class discr_GymStyle_Robot(DiscretizedActionWrapper):
    def __init__(self, n_bins_act=10, low_act=None, high_act=None, **kwargs):
        
        env = RobotEnv(**kwargs)
        super().__init__(env, n_bins_act, low_act, high_act)
        
        
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
        
        
#%%
if __name__ == "__main__":
    #env = RobotEnv()
    #my_env = GymStyle_Robot(n_bins_act=4, normalize_obs_state = False)
    my_env = discr_GymStyle_Platoon(n_bins_act=4)
