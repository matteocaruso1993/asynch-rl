#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:50:55 2020

@author: Enrico Regolin
"""

# tester robot
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

#####
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-v", "--version", dest="net_version", type = int, default= 903 , help="training version")

parser.add_argument("-i", "--iter"   , dest="iteration"  , type = int, default= 4200 , help="iteration")

parser.add_argument("-sim", "--simulate"   , dest="simulate"  , type = bool, default= True , help="simulate instance")

parser.add_argument("-d", "--difficulty"   , dest="difficulty"  , type = int, default= 0 , help="difficulty")

args = parser.parse_args()
################


# df loader
#net_version = 902
#iteration   = 27


# generate proper discretized bins structure

def main(net_version = 100, iteration = 2, simulate = False, difficulty = 0):

    ################
    env_type = 'RobotEnv' 
    model_type = 'ConvModel'
    #rl_mode = 'AC'
    
    overwrite_params = ['rewards', 'rl_mode', 'share_conv_layers', 'n_frames' , 'layers_width', 'map_output', 'normalize_layers']
        
    my_dict = load_train_params(env_type, model_type, overwrite_params, net_version)
    local_vars = locals()
    
    for i,par in enumerate(overwrite_params):
        #exec(par + " =  my_dict['"  + par + "']", None, )
        local_vars[par] = my_dict[par]
    del( overwrite_params, my_dict)
    
    
    
    ################
    """
    my_vars = locals().copy()
    for v in my_vars:
        print(v)
    """
    import inspect
    inspect.signature(Multiprocess_RL_Environment.__init__)
    
    
    rl_env = Multiprocess_RL_Environment(env_type, model_type, net_version, rl_mode=local_vars['rl_mode'] , ray_parallelize=False, \
                                         move_to_cuda=False, n_frames = local_vars['n_frames'], show_rendering = True, discr_env_bins=2,\
                                        difficulty= difficulty, map_output = local_vars['map_output'], \
                                          layers_width = local_vars['layers_width'], normalize_layers = local_vars['normalize_layers']  ) #, \
                                      #    #replay_memory_size = 500, N_epochs = 100)
    
    
    print(f'rl mode : {rl_env.rl_mode}')
    
    rl_env.save_movie = False
    rl_env.live_plot = False
    # always update agents params after rl_env params are changed
    rl_env.updateAgentsAttributesExcept('env')
    
    rl_env.load(iteration)
    #rl_env.load(320)
    
    rl_env.print_NN_parameters_count()
    
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
    if simulate:
        agent = rl_env.sim_agents_discr[0]
        #agent.live_plot = True
        
        #agent.max_steps_single_run = 20000
        
        #
        agent.movie_frequency = 1
        #agent.tot_iterations = 10000
        agent.tot_iterations = 300
        agent.max_n_single_runs = 5
        sim_log, single_runs , successful_runs,_, pg_info = agent.run_synch(use_NN = True, test_qv = False)
    
    

#%%
################################################################

if __name__ == "__main__":
    
    main(net_version = args.net_version, iteration = args.iteration, simulate = args.simulate, difficulty = args.difficulty)

    current_folder = os.path.abspath(os.path.dirname(__file__))
    clear_pycache(current_folder)

