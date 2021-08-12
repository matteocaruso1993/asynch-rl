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

parser.add_argument("-v", "--version", dest="net_version", type = int, default= 200 , help="training version")

parser.add_argument("-i", "--iter"   , dest="iteration"  , type = int, default= -1 , help="iteration")

parser.add_argument("-sim", "--simulate"   , dest="simulate"  , type=lambda x: (str(x).lower() in ['true','1', 'yes']), default= False , help="simulate instance")

parser.add_argument("-d", "--difficulty"   , dest="difficulty"  , type = int, default= 2 , help="difficulty")

parser.add_argument("-s", "--save-movie"   , dest="save_movie"  , type=lambda x: (str(x).lower() in ['true','1', 'yes']), default= True , help="save movie")

parser.add_argument("-e", "--eps-format"   , dest="eps_format"  , type=lambda x: (str(x).lower() in ['true','1', 'yes']), default= False , help="eps_format")

parser.add_argument("-dt", "--step-size"   , dest="step_size"  , type = float, default= 0.4 , help="simulation step size")

args = parser.parse_args()
################


# generate proper discretized bins structure

def main(net_version = 100, iteration = 2, simulate = False, difficulty = 0, save_movie = False, eps_format = False, step_size = 0.4):

    ################
    env_type = 'RobotEnv' 
    model_type = 'ConvModel'
    #rl_mode = 'AC'
    
    overwrite_params = ['rewards', 'rl_mode', 'share_conv_layers', 'n_frames' ,\
                        'layers_width', 'map_output', 'normalize_layers', \
                            'val_frequency']
        
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
                                          layers_width = local_vars['layers_width'], normalize_layers = local_vars['normalize_layers'] ,\
                                              rewards=local_vars['rewards'], val_frequency=local_vars['val_frequency'], step_size = step_size) #, \
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
        fig0, fig  = rl_env.plot_training_log(0, qv_loss_log = False, \
                                                      pg_loss_log = False, save_fig = save_movie, eps_format=eps_format)
            
    except Exception:
        print('incomplete data for plot generation')
    
    


    
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
        agent.max_steps_single_run = int(1000*0.4/step_size)
        
        #
        agent.movie_frequency = 1
        #agent.tot_iterations = 10000
        agent.tot_iterations = int(500*0.4/step_size)
        agent.max_n_single_runs = 5

        if save_movie:
            rl_env.update_net_name()
            agent.net_name = rl_env.net_name
            agent.save_movie = True
            agent.tot_iterations = int(2500*0.4/step_size)
            agent.max_n_single_runs = 10
        
        sim_log, single_runs , successful_runs,_,_, pg_info = agent.run_synch(use_NN = True, test_qv = False)
        
        if 'fig0' in locals():
            fig.waitforbuttonpress(20)
            fig0.waitforbuttonpress(20)
            #fig_st.waitforbuttonpress(20)

    """
    stats = []
    n_samples = 40
    for i in range(1,n_samples):
        pctg_success = (round(100*rl_env.traj_stats[-i].count('success')/len(rl_env.traj_stats[-i])))
        stats.append(pctg_success)
    print(f'success percentage last {n_samples}: {sum(stats)/len(stats)}%')
    """


#%%
################################################################

if __name__ == "__main__":
    
    main(net_version = args.net_version, iteration = args.iteration, simulate = args.simulate, \
         difficulty = args.difficulty, save_movie=args.save_movie, eps_format=args.eps_format, step_size = args.step_size)

    current_folder = os.path.abspath(os.path.dirname(__file__))
    clear_pycache(current_folder)

