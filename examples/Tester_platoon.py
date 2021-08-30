#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:01:24 2020

@author: Enrico Regolin
"""


# tester platoon

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

parser.add_argument("-v", "--version", dest="net_version", type = int, default= 31 , help="training version")

parser.add_argument("-i", "--iter"   , dest="iteration"  , type = int, default= -1 , help="iteration")

parser.add_argument("-sim", "--simulate"   , dest="simulate"  , type=lambda x: (str(x).lower() in ['true','1', 'yes']), default= True , help="simulate instance")

parser.add_argument("-d", "--difficulty"   , dest="difficulty"  , type = int, default= 1 , help="difficulty")

parser.add_argument("-s", "--save-movie"   , dest="save_movie"  , type=lambda x: (str(x).lower() in ['true','1', 'yes']), default= False , help="save movie")

parser.add_argument("-e", "--eps-format"   , dest="eps_format"  , type=lambda x: (str(x).lower() in ['true','1', 'yes']), default= False , help="eps_format")

args = parser.parse_args()
################


# generate proper discretized bins structure

def main(net_version = 100, iteration = 2, simulate = False, difficulty = 0, save_movie = False, eps_format = False):


    
    # generate proper discretized bins structure
    ################
    env_type = 'Platoon' 
    model_type = 'LinearModel'
    overwrite_params = ['layers_width','discr_act_bins', 'n_gears', 'rewards', 'rl_mode', 'normalize_layers', 'val_frequency']
        
    my_dict = load_train_params(env_type, model_type, overwrite_params, net_version)
    local_vars = locals()
    
    for i,par in enumerate(overwrite_params):
        #exec(par + " =  my_dict['"  + par + "']", None, )
        local_vars[par] = my_dict[par]
    del( overwrite_params, my_dict)
    

    import inspect
    inspect.signature(Multiprocess_RL_Environment.__init__)
    
    env_options = {'n_gears' : local_vars['n_gears']}
    
    #discr_act_bins = local_vars['discr_act_bins']
    
    #if change_gears:
    #    discr_act_bins.append(1)
   
    
    rl_env = Multiprocess_RL_Environment(env_type, model_type, net_version, rl_mode=local_vars['rl_mode'] , ray_parallelize=False, \
                                         move_to_cuda=False, show_rendering = True, discr_env_bins=local_vars['discr_act_bins'],\
                                        difficulty= difficulty, layers_width = local_vars['layers_width'], normalize_layers = local_vars['normalize_layers'] ,\
                                              rewards=local_vars['rewards'], val_frequency=local_vars['val_frequency'], env_options = env_options) #, \
    
    
    rl_env.save_movie = False
    rl_env.live_plot = False
    # always update agents params after rl_env params are changed
    rl_env.updateAgentsAttributesExcept('env')
    
    rl_env.load(iteration)
    #rl_env.load(320)
    
    rl_env.plot_training_log(1)
    
    
    #%%
    
    
    agent = rl_env.sim_agents_discr[0]
    
    #agent.max_steps_single_run = 5000
    #
    agent.movie_frequency = 1
    agent.tot_iterations = 500
    agent.max_n_single_runs = 1
    agent.env.env.sim_length_max = 200
    
    
    sim_log, single_runs , successful_runs,_,_, pg_info = agent.run_synch(use_NN = True, test_qv = False)
    
    agent.env.env.plot_graphs()



#%%
################################################################

if __name__ == "__main__":
    
    main(net_version = args.net_version, iteration = args.iteration, simulate = args.simulate, \
         difficulty = args.difficulty, save_movie=args.save_movie, eps_format=args.eps_format)

    current_folder = os.path.abspath(os.path.dirname(__file__))
    clear_pycache(current_folder)
