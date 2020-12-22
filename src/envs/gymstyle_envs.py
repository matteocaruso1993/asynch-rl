# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:24:29 2020

@author: Enrico Regolin
"""

#%%

from envs.cartpole_env.CartPole_env import CartPoleEnv
from envs.platoon_env.Platooning_env import PlatoonEnv
from envs.robot_env.Robot_env import RobotEnv

from envs.wrappers import DiscretizedActionWrapper,DiscretizedObservationWrapper, ContinuousHybridActionWrapper


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
        
        
#%%
if __name__ == "__main__":
    #env = RobotEnv()
    #my_env = GymStyle_Robot(n_bins_act=4, normalize_obs_state = False)
    my_env = discr_GymStyle_Platoon(n_bins_act=4)
