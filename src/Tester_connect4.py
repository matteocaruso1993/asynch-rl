#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:06:26 2021

@author: Enrico Regolin
"""


# tester robot
import matplotlib.pyplot as plt
from rl.utilities import clear_pycache, load_train_params

from rl.rl_env import Multiprocess_RL_Environment

import sys
import psutil
import time

import numpy as np
import os

import asyncio

# df loader
net_version = 1
iteration = 20

# generate proper discretized bins structure
################
env_type = 'Connect4' 
model_type = 'LinearModel'
overwrite_params = ['layers_width', 'rewards']
rl_mode = 'AC'

my_dict = load_train_params(env_type, model_type, overwrite_params, net_version)
for i,par in enumerate(overwrite_params):
    exec(par + " =  my_dict['"  + par + "']")
del( overwrite_params, my_dict)

################

rl_env = Multiprocess_RL_Environment(env_type, model_type, net_version,rl_mode = rl_mode, ray_parallelize=False, rewards = rewards, \
                                     move_to_cuda=False, n_frames = 0, show_rendering = True) #, \
                                      #replay_memory_size = 500, N_epochs = 100)


rl_env.save_movie = False
rl_env.live_plot = False
# always update agents params after rl_env params are changed
rl_env.updateAgentsAttributesExcept('env')

rl_env.load(iteration, load_memory = False)
#rl_env.load(320)

rl_env.plot_training_log(1)


if hasattr(rl_env, 'val_history'):

    
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

#%%


agent = rl_env.sim_agents_discr[0]

#agent.max_steps_single_run = 5000
#
agent.movie_frequency = 1
agent.tot_iterations = 400
agent.max_n_single_runs = 10


sim_log, single_runs , successful_runs= agent.run_synch(use_NN = True, test_qv = True)

#agent.env.env.plot_graphs()


#%%

current_folder = os.path.abspath(os.path.dirname(__file__))
clear_pycache(current_folder)
