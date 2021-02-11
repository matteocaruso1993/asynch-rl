#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 12:00:07 2021

@author: Enrico Regolin
"""

#%%

from rl.rl_env import Multiprocess_RL_Environment
from rl.utilities import clear_pycache, store_train_params, load_train_params


import sys
import psutil
import time
import os
import ray

import numpy as np

#####
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-rl", "--rl-mode", dest="rl_mode", type=str, default='AC', help="RL mode (AC, DQL, parallelAC)")

parser.add_argument("-i", "--iter", dest="n_iterations", type = int, default= 5 , help="number of training iterations")

parser.add_argument("-p", "--parallelize", dest="ray_parallelize", type=bool, default=False,
                    help="ray_parallelize bool")

parser.add_argument("-a", "--agents-number", dest="agents_number", type=int, default=20,
                    help="Number of agents to be used")


parser.add_argument("-l", "--load-iteration", dest="load_iteration", type=int, default=0,
                    help="start simulations and training from a given iteration")

parser.add_argument("-m", "--memory-size", dest="replay_memory_size", type=int, default= 10000,
                    help="Replay Memory Size")

parser.add_argument("-v", "--net-version", dest="net_version", type=int, default=100,
                    help="net version used")

#####
parser.add_argument("-tot", "--tot-iterations", dest="tot_iterations", type=int, default= 1000,
                    help="Max n. iterations each agent runs during simulation. Influences the level of exploration which is reached by PG algorithm")

parser.add_argument("-sim", "--sim-length-max", dest="sim_length_max", type=int, default=720,
                    help="Length of one run")

parser.add_argument("-mt", "--memory-turnover-ratio", dest="memory_turnover_ratio", type=float, default=.5,
                    help="Ratio of Memory renewed at each iteration")

parser.add_argument("-lr", "--learning-rate", dest="learning_rate", nargs="*", type=float, default=[1e-4, 1e-4],
                    help="NN learning rate")

parser.add_argument("-e", "--epochs-training", dest="n_epochs", type=int, default= 400 , help="Number of epochs per training iteration")

parser.add_argument("-mb", "--minibatch-size", dest="minibatch_size",  nargs="*", type=int, default= 512,
                    help="Size of the minibatches used for QV training")

parser.add_argument("-y", "--epsilon", dest="epsilon", nargs=2, type=float, default=[0.9995 , 0.2],
                    help="two values: initial epsilon, final epsilon")

parser.add_argument("-yd", "--epsilon-decay", dest="epsilon_decay", type=float, default=0.9,
                    help="annealing factor of epsilon")

parser.add_argument("-vf", "--validation-frequency", dest="val_frequency", type=int, default=20,
                    help="model is validated every -vf iterations")

parser.add_argument("-ro", "--reset-optimizer", dest="reset_optimizer", type=bool, default=False,
                    help="reset optimizer")

parser.add_argument("-fr", "--frames-number", dest="n_frames", type=int, default=240,
                    help="number of frames considered for convolutional network")

parser.add_argument("-scl", "--share-conv-layers", dest="share_conv_layers", type=bool, default=True,
                    help="Flag to share Convolutional Layers between Actor and Critic")

parser.add_argument("-g", "--gamma", dest="gamma", type=float, default=0.95, help="GAMMA parameter in QV learning")

parser.add_argument("-b", "--beta", dest="beta", nargs=2, type=float, default= 0.1 , help="BETA parameter for entropy in PG learning")

parser.add_argument("-cadu", "--continuous-advantage-update", dest="continuous_qv_update", type=bool, default=False, 
                    help="latest QV model is always used for Advanatge calculation")

parser.add_argument( "-rw", "--rewards",  nargs="*",  dest = "rewards_list", type=int, default=[100, 1] )

parser.add_argument("-nc", "--no-cuda", dest="no_cuda", type=bool, default=False, help="avoid using cuda")

args = parser.parse_args()

#####

num_cpus = psutil.cpu_count(logical=False)
n_agents = 2*num_cpus -2

def main(net_version = 0, n_iterations = 2, ray_parallelize = False,  difficulty = 0,\
         load_iteration = -1, agents_number = n_agents, learning_rate= 0.001,\
             n_epochs = 400, replay_memory_size = 5000, epsilon = [.9, 0.1], ctrlr_probability = 0, sim_length_max = 100, \
        epsilon_annealing_factor = 0.95,  ctrlr_prob_annealing_factor = 0.9 , mini_batch_size = 64, \
            memory_turnover_ratio = 0.1, val_frequency = 10, rewards = np.ones(4), reset_optimizer = False,
            share_conv_layers = False, n_frames = 4, rl_mode = 'DQL', beta = 0.001, \
                gamma = 0.99,  continuous_qv_update = False, tot_iterations = 400, no_cuda = False):


    function_inputs = locals().copy()
    
    env_type = 'Frx' 
    model_type = 'ConvFrxModel'
    overwrite_params = ['rewards', 'share_conv_layers', 'n_frames' ]
    
    # trick used to resume epsilon status if not declared explicitly
    if epsilon[0] == -1:
        epsilon[0] = 0.9
        resume_epsilon = True
    else:
        resume_epsilon = False

        
    # initialize required net and model parameters if loading from saved values
    if load_iteration != 0:

        storage_path = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) ,"Data" , \
                                env_type, model_type+str(net_version) )

        if os.path.isfile(os.path.join(storage_path,'train_params.txt')):
            my_dict = load_train_params(env_type, model_type, overwrite_params, net_version)
            for i,par in enumerate(overwrite_params):
                exec(par + " =  my_dict['"  + par + "']")
            del( overwrite_params, my_dict)

    # import ray
    if ray_parallelize:
        # Start Ray.
        try:
            ray.shutdown()
        except Exception:
            print('ray not active')
        ray.init()

    #single_agent_min_iterations = round(memory_turnover_ratio*replay_memory_size / (agents_number * 20) )
    
    rl_env = Multiprocess_RL_Environment(env_type , model_type , net_version , rl_mode = rl_mode, \
                        ray_parallelize=ray_parallelize, n_frames = n_frames, \
                        replay_memory_size = replay_memory_size, n_agents = agents_number,\
                        tot_iterations = tot_iterations, \
                        epsilon_annealing_factor=epsilon_annealing_factor,\
                        N_epochs = n_epochs, epsilon = epsilon[0] , epsilon_min = epsilon[1] , rewards = rewards, \
                        mini_batch_size = mini_batch_size, \
                        share_conv_layers = share_conv_layers, move_to_cuda = not no_cuda, \
                        difficulty = difficulty, learning_rate = learning_rate, sim_length_max = sim_length_max, \
                        memory_turnover_ratio = memory_turnover_ratio, val_frequency = val_frequency ,\
                        gamma = gamma, beta_PG = beta , continuous_qv_update = continuous_qv_update)


    rl_env.resume_epsilon = resume_epsilon

    rl_env.save_movie = False
    rl_env.live_plot = False
    # always update agents params after rl_env params are changed
    rl_env.updateAgentsAttributesExcept('env')

    # launch env
    time_init = time.time()
    
    # second run is to test parallel computation fo simulations and NN update
    if load_iteration != 0:
        rl_env.load( load_iteration)
        
    else:
        store_train_params(rl_env, function_inputs)
        
    rl_env.runSequence(n_iterations, reset_optimizer=reset_optimizer) 

    return rl_env


################################################################

if __name__ == "__main__":
    
    env = main(net_version = args.net_version, n_iterations = args.n_iterations, ray_parallelize= args.ray_parallelize, \
               load_iteration =args.load_iteration, replay_memory_size = args.replay_memory_size, \
               agents_number = args.agents_number, memory_turnover_ratio = args.memory_turnover_ratio, \
               n_epochs = args.n_epochs, epsilon = args.epsilon, epsilon_annealing_factor = args.epsilon_decay, \
               mini_batch_size = args.minibatch_size,  learning_rate = args.learning_rate, \
               sim_length_max = args.sim_length_max, val_frequency = args.val_frequency, share_conv_layers = args.share_conv_layers, \
               rewards = args.rewards_list, reset_optimizer = args.reset_optimizer, rl_mode = args.rl_mode, \
               beta = args.beta, gamma = args.gamma, continuous_qv_update = args.continuous_qv_update,\
               tot_iterations = args.tot_iterations , n_frames = args.n_frames, no_cuda = args.no_cuda)

    current_folder = os.path.abspath(os.path.dirname(__file__))
    clear_pycache(current_folder)

