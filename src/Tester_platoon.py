#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:01:24 2020

@author: rodpod21
"""


# tester robot
import matplotlib.pyplot as plt


from RL.RL_manager import Multiprocess_RL_Environment

import sys
import psutil
import time

import numpy as np
import os

import asyncio

# df loader
net_version = 1
iteration = 500

# generate proper discretized bins structure
################
my_dict = {}
overwrite_params = ['layers_width','discr_act_bins', 'rewards']

storage_path = os.path.join( os.path.dirname(os.path.abspath(__file__)) ,"Data" , \
                        'Platoon', 'LinearModel'+str(net_version) )

with open(os.path.join(storage_path,'train_params.txt'), 'r+') as f:
    Lines = f.readlines() 
    
for line in Lines: 
    for var in overwrite_params:
        if var in line:
            my_dict.__setitem__(var , eval(line[line.find(':')+1 : line.find('\n')])) 
 
layers_width = my_dict['layers_width']
discr_act_bins = my_dict['discr_act_bins']
rewards = my_dict['rewards']
del( overwrite_params, Lines, storage_path, my_dict)
################

dqn_env = Multiprocess_RL_Environment('Platoon', 'LinearModel', net_version, ray_parallelize=False, difficulty=1, rewards = rewards, \
                                     move_to_cuda=False, n_frames = 0, discr_env_bins=discr_act_bins, save_override= False, test_mode = True ) #, \
                                      #replay_memory_size = 500, N_epochs = 100)


dqn_env.save_movie = False
dqn_env.live_plot = False
# always update agents params after dqn_env params are changed
dqn_env.updateAgentsAttributesExcept('env')

dqn_env.load(iteration, load_memory = False)
#dqn_env.load(320)

dqn_env.plot_training_log(1)


if hasattr(dqn_env, 'val_history'):

    
    # 0) iteration ---- 1) average duration   ---- 2)average single run reward   ---- 3) average loss
    
    fig_val1 = plt.figure()
    ax1 = fig_val1.add_subplot(4,1,1)
    ax2 = fig_val1.add_subplot(4,1,2)
    ax3 = fig_val1.add_subplot(4,1,3)
    ax4 = fig_val1.add_subplot(4,1,4)

    ax1.plot(dqn_env.val_history[:,0], dqn_env.val_history[:,2])
    ax1.legend(['total runs'])
        
    ax2.plot(dqn_env.val_history[:,0], dqn_env.val_history[:,3])
    ax2.legend(['average duration'])
    
    ax3.plot(dqn_env.val_history[:,0], dqn_env.val_history[:,4])
    ax3.legend(['average cum reward'])

    ax4.plot(dqn_env.val_history[:,0], dqn_env.val_history[:,1])
    ax4.legend(['successful runs ratio'])

#%%


agent = dqn_env.sim_agents[0]

#agent.max_steps_single_run = 5000
#
agent.movie_frequency = 1
agent.tot_iterations = 500
agent.max_n_single_runs = 1
agent.env.env.sim_length_max = 200


sim_log, single_runs , successful_runs= agent.run_synch(use_NN = True)

agent.env.env.plot_graphs()