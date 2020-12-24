#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:51:07 2020

@author: Enrico Regolin
"""
#%%
# compiling time debug
import cProfile
import pstats
import io


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

parser.add_argument("-i", "--iter", dest="n_iterations", type = int, default= 10, help="number of training iterations")

parser.add_argument("-p", "--parallelize", dest="ray_parallelize", type=bool, default=False,
                    help="ray_parallelize bool")

parser.add_argument("-d","--difficulty", dest = "difficulty", type=int, default=0, help = "task degree of difficulty")

"""
parser.add_argument("-r", "--record-time", dest="record_computing_time", type=bool, default=False,
                    help="flag to record computation time of each function")
"""

parser.add_argument("-l", "--load-iteration", dest="load_iteration", type=int, default=0,
                    help="start simulations and training from a given iteration")

parser.add_argument("-sim", "--sim-length-max", dest="sim_length_max", type=int, default=100,
                    help="Length of one successful run in seconds")

parser.add_argument("-m", "--memory-size", dest="replay_memory_size", type=int, default=2000,
                    help="Replay Memory Size")

parser.add_argument("-mt", "--memory-turnover-ratio", dest="memory_turnover_ratio", type=float, default=.5,
                    help="Ratio of Memory renewed at each iteration")

parser.add_argument("-a", "--agents-number", dest="agents_number", type=int, default=2,
                    help="Number of agents to be used")

parser.add_argument("-lr", "--learning-rate", dest="learning_rate", type=float, default=1e-4,
                    help="NN learning rate")

parser.add_argument("-e", "--epochs-training", dest="n_epochs", type=int, default=500,
                    help="Number of epochs per training iteration")

parser.add_argument("-mb", "--minibatch-size", dest="minibatch_size", type=int, default=128,
                    help="Size of the minibatches used for training")

parser.add_argument("-y", "--epsilon", dest="epsilon", type=float, default=-1,
                    help="initial epsilon")

parser.add_argument("-yd", "--epsilon-decay", dest="epsilon_decay", type=float, default=0.995,
                    help="annealing factor of epsilon")

parser.add_argument("-c", "--controller-probability", dest="ctrl_prob", type=float, default=0.4,
                    help="probability of control action being decided by conventional controller")

parser.add_argument("-cd", "--controller-probability-decay", dest="ctrl_prob_decay", type=float, default=0.9,
                    help="annealing factor of control probability")

parser.add_argument("-vf", "--validation-frequency", dest="val_frequency", type=int, default=10,
                    help="model is validated every -vf iterations")

parser.add_argument("-v", "--net-version", dest="net_version", type=int, default=100,
                    help="net version used")

parser.add_argument("-gc", "--change-gears", dest="change_gears", type=bool, default=False,
                    help="change gears")

parser.add_argument("-ro", "--reset-optimizer", dest="reset_optimizer", type=bool, default=False,
                    help="reset optimizer")

parser.add_argument("-rl", "--rl-mode", dest="rl_mode", type=str, default='AC',
                    help="RL mode (AC, DQL)")

"""
parser.add_argument("-pu", "--pg-partial-update", dest="pg_partial_update", type=bool, default=False,
                    help="Flag to update only partially the PG model (non updated layers are shared with QV model)")
"""
parser.add_argument(
  "-ll", "--layers-list",  nargs="*",  # 0 or more values expected => creates a list
  dest = "layers_list", type=int, default=[20,20, 2],  # default if nothing is provided
)

parser.add_argument(
  "-rw", "--rewards",  nargs="*",  # 0 or more values expected => creates a list
  dest = "rewards_list", type=int, default=[200,50, 1,1],  # default if nothing is provided
)

parser.add_argument(
  "-dab", "--discrete-action-bins",  nargs="*",  # 0 or more values expected => creates a list
  dest = "discr_act_bins", type=int, default=[10,1],  # default if nothing is provided
)

args = parser.parse_args()

#####

num_cpus = psutil.cpu_count(logical=False)
n_agents = 2*num_cpus -2

def main(net_version = 0, n_iterations = 2, ray_parallelize = False, record_computing_time = False, difficulty = 0,\
         load_iteration = -1, agents_number = n_agents, learning_rate= 0.001,\
             n_epochs = 400, replay_memory_size = 5000, epsilon = .9, ctrlr_probability = 0, sim_length_max = 100, \
        epsilon_annealing_factor = 0.95,  ctrlr_prob_annealing_factor = 0.9 , mini_batch_size = 64, \
            memory_turnover_ratio = 0.1, val_frequency = 10, layers_width= (5,5), discr_act_bins = (20, 1) ,\
                rewards = np.ones(5), change_gears = False, reset_optimizer = False, rl_mode = 'DQL'):

    function_inputs = locals().copy()
    
    env_type = 'Platoon' 
    model_type = 'LinearModel'
    overwrite_params = ['layers_width','discr_act_bins', 'change_gears']
    
    # trick used to resume epsilon status if not declared explicitly
    if epsilon == -1:
        epsilon = 0.9
        resume_epsilon = True
    else:
        resume_epsilon = False

    # initialize required net and model parameters if loading from saved values
    if load_iteration != 0:
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
        

    env_options = {'change_gears' : change_gears}
    
    single_agent_min_iterations = round(memory_turnover_ratio*replay_memory_size / (agents_number * 20) )
    
    rl_env = Multiprocess_RL_Environment(env_type, model_type , net_version ,rl_mode = rl_mode ,n_agents = agents_number, \
                                         ray_parallelize=ray_parallelize, move_to_cuda=True, n_frames = 1, \
                                         replay_memory_size = replay_memory_size, N_epochs = n_epochs, \
                                         epsilon = epsilon , tot_iterations = single_agent_min_iterations, \
                                         ctrlr_probability = ctrlr_probability, epsilon_annealing_factor=epsilon_annealing_factor,\
                                         ctrlr_prob_annealing_factor = ctrlr_prob_annealing_factor, discr_env_bins = discr_act_bins , \
                                         mini_batch_size = mini_batch_size, difficulty = difficulty,\
                                         learning_rate = learning_rate, sim_length_max = sim_length_max, \
                                         memory_turnover_ratio = memory_turnover_ratio, val_frequency = val_frequency ,\
                                         layers_width = layers_width, rewards = rewards, env_options = env_options)
        
    rl_env.resume_epsilon = resume_epsilon


    #rl_env.movie_frequency = 2
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


    pr = cProfile.Profile()
    pr.enable()

    rl_env.runSequence(n_iterations, reset_optimizer=reset_optimizer) 

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()
    with open('duration_test.txt', 'w+') as f:
        f.write(s.getvalue())

    
    return rl_env


############################################################################


if __name__ == "__main__":
    
    env = main(net_version = args.net_version, n_iterations = args.n_iterations, ray_parallelize= args.ray_parallelize, \
               load_iteration =args.load_iteration, replay_memory_size = args.replay_memory_size, agents_number = args.agents_number, \
               n_epochs = args.n_epochs, epsilon = args.epsilon, ctrlr_probability = args.ctrl_prob, \
               epsilon_annealing_factor = args.epsilon_decay, ctrlr_prob_annealing_factor = args.ctrl_prob_decay, \
               mini_batch_size = args.minibatch_size,  learning_rate = args.learning_rate, \
               sim_length_max = args.sim_length_max, difficulty = args.difficulty, \
               val_frequency = args.val_frequency, memory_turnover_ratio = args.memory_turnover_ratio, \
               layers_width= args.layers_list, discr_act_bins = args.discr_act_bins, rewards = args.rewards_list, \
               change_gears = args.change_gears, reset_optimizer = args.reset_optimizer, rl_mode = args.rl_mode)
        
    current_folder = os.path.abspath(os.path.dirname(__file__))
    clear_pycache(current_folder)

    
