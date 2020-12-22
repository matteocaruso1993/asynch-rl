#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:50:55 2020

@author: rodpod21
"""

# tester robot
import matplotlib.pyplot as plt

from DQN_env import Multiprocess_DQN_Environment

import sys
import psutil
import time

import numpy as np

import asyncio

# df loader

dqn_env = Multiprocess_DQN_Environment('RobotEnv', 'ConvModel', 3, ray_parallelize=False, difficulty=0, \
                                     move_to_cuda=False, n_frames = 4, discr_env_bins=2, save_override= False, test_mode = True ) #, \
                                      #replay_memory_size = 500, N_epochs = 100)


dqn_env.save_movie = False
dqn_env.live_plot = False
# always update agents params after dqn_env params are changed
dqn_env.updateAgentsAttributesExcept('env')

dqn_env.load(2000, load_memory = False)
#dqn_env.load(320)

dqn_env.plot_training_log()


if hasattr(dqn_env, 'val_history'):

    """
    # while file is not fixed
    mask = np.ones(dqn_env.val_history.shape[0], dtype = bool)
    mask[16:19] = False
    dqn_env.val_history = dqn_env.val_history[mask]
    """
    
    # 0) iteration ---- 1) average duration   ---- 2)average single run reward   ---- 3) average loss
    
    fig_val1 = plt.figure()
    ax1 = fig_val1.add_subplot(3,1,1)
    ax2 = fig_val1.add_subplot(3,1,2)
    ax3 = fig_val1.add_subplot(3,1,3)
    ax1.plot(dqn_env.val_history[:,0], dqn_env.val_history[:,1])
    ax1.legend(['average loss'])
    
    
    ax2.plot(dqn_env.val_history[:,0], dqn_env.val_history[:,2])
    ax2.legend(['average duration'])
    
    ax3.plot(dqn_env.val_history[:,0], dqn_env.val_history[:,3])
    ax3.legend(['average cum reward'])


#%%
# script to clean up val hist

"""
import numpy as np

mask = np.ones(dqn_env.val_history.shape[0], dtype = bool)
mask[16:19] = False

dqn_env.val_history = dqn_env.val_history[mask]
"""

"""
#save it afterwards
import os
#path = os.path.dirname(os.path.abspath(__file__))
path = os.getcwd()
val_history_file = os.path.join(path, 'val_history.npy')
np.save(val_history_file, dqn_env.val_history )
"""


#"""
#%%
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

def is_Ipython():
    try:
        get_ipython()
        return True
    except Exception:
        return False

#%%

agent = dqn_env.sim_agents[0]

#agent.max_steps_single_run = 20000

#
agent.movie_frequency = 1
#agent.tot_iterations = 10000
agent.tot_iterations = 500
agent.max_n_single_runs = 20

async def synch_tester(save_movie = False):
    agent.save_movie = save_movie
    task = asyncio.create_task(agent.run(use_NN = True))
    await task
    
#print(is_Ipython())
    
if not is_Ipython():
    asyncio.run(synch_tester(save_movie = True))
else:  
    agent.live_plot = True
    loop = asyncio.get_event_loop()
    try:
        asyncio.ensure_future(synch_tester())
        loop.run_forever()
    except KeyboardInterrupt:
        pass


# doesn't work in Ipython!

#agent.env.env.cartpole.plot_graphs(dt=agent.env.env.dt, save = False, no_norm = False, ml_ctrl = True)
#"""