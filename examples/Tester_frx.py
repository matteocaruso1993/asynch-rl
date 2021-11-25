#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:54:46 2021

@author: rodpod21
"""


# tester frx
import os, sys

from asynch_rl.rl.rl_env    import Multiprocess_RL_Environment
from asynch_rl.rl.utilities import clear_pycache, load_train_params

import sys
import psutil
import time

import matplotlib.pyplot as plt
import numpy as np
import os

import asyncio

# df loader
net_version = 100
iteration = -1

# generate proper discretized bins structure
################
env_type = 'Frx' 
model_type = 'ConvFrxModel'
rl_mode = 'AC'

overwrite_params = ['rewards', 'rl_mode']

my_dict = load_train_params(env_type, model_type, overwrite_params, net_version)
for i,par in enumerate(overwrite_params):
    exec(par + " =  my_dict['"  + par + "']")
del( overwrite_params, my_dict)


################

rl_env = Multiprocess_RL_Environment(env_type, model_type, net_version,rl_mode = rl_mode, ray_parallelize=False, \
                                     move_to_cuda=False, show_rendering = True, n_frames = 240) #, \
                                      #replay_memory_size = 500, N_epochs = 100)


rl_env.save_movie = False
rl_env.live_plot = False
# always update agents params after rl_env params are changed
rl_env.updateAgentsAttributesExcept('env')

rl_env.load(iteration)
#rl_env.load(320)

try:
    rl_env.plot_training_log(1, qv_loss_log = True, pg_loss_log = True)
except Exception:
    print('incomplete data for plot generation')


try:
    if hasattr(rl_env, 'val_history'):
        
        if rl_env.val_history is not None:
            # 0) iteration ---- 1) average duration   ---- 2)average single run reward   ---- 3) average loss
            
            fig_val1 = plt.figure()
            ax1 = fig_val1.add_subplot(4,1,1)
            ax2 = fig_val1.add_subplot(4,1,2)
            ax3 = fig_val1.add_subplot(4,1,3)
            ax4 = fig_val1.add_subplot(4,1,4)
        
            ax1.plot(rl_env.val_history[:,0], rl_env.val_history[:,2])
            ax1.legend(['total runs'])
                
            ax2.plot(rl_env.val_history[:,0], rl_env.val_history[:,3])
            ax2.legend(['average duration'])
            
            ax3.plot(rl_env.val_history[:,0], rl_env.val_history[:,4])
            ax3.legend(['average cum reward'])
        
            ax4.plot(rl_env.val_history[:,0], rl_env.val_history[:,1])
            ax4.legend(['successful runs ratio'])
except Exception:
    pass

#%%
# script to clean up val hist

"""
import numpy as np

mask = np.ones(rl_env.val_history.shape[0], dtype = bool)
mask[16:19] = False

rl_env.val_history = rl_env.val_history[mask]
"""

"""
#save it afterwards
import os
#path = os.path.dirname(os.path.abspath(__file__))
path = os.getcwd()
val_history_file = os.path.join(path, 'val_history.npy')
np.save(val_history_file, rl_env.val_history )
"""


#"""

#%%

agent = rl_env.sim_agents_discr[0]

#agent.max_steps_single_run = 20000

#
agent.movie_frequency = 1
#agent.tot_iterations = 10000
agent.tot_iterations = 1000
agent.max_n_single_runs = 5
sim_log, single_runs , successful_runs, _, pg_info = agent.run_synch(use_NN = True, test_qv = False)

agent.env.env.plot_graphs()


#%%

current_folder = os.path.abspath(os.path.dirname(__file__))
clear_pycache(current_folder)


"""
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
"""

# doesn't work in Ipython!

#agent.env.env.cartpole.plot_graphs(dt=agent.env.env.dt, save = False, no_norm = False, ml_ctrl = True)
#"""