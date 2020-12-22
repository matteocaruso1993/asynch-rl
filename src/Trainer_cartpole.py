#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:11:08 2020

@author: rodpod21
"""

from DQN_env import Multiprocess_DQN_Environment

import sys
import psutil
import time


#####
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", "--iter", dest="n_iterations", type = int, default=1, help="number of training iterations")
parser.add_argument("-p", "--parallelize", dest="ray_parallelize", type=bool, default=False,
                    help="ray_parallelize bool")

parser.add_argument("-d","--difficulty", dest = "difficulty", type=int, default=0, help = "task degree of difficulty")

parser.add_argument("-r", "--record-time", dest="record_computing_time", type=bool, default=False,
                    help="flag to record computation time of each function")
parser.add_argument("-l", "--load-iteration", dest="load_iteration", type=int, default=-1,
                    help="start simulations and training from a given iteration")
parser.add_argument("-s", "--save-override", dest="save_override", type=bool, default=False,
                    help="override existing saved version")

parser.add_argument("-sf", "--save-frequency", dest="save_frequency", type=int, default=10,
                    help="saving frequency (every -sf iterations)")

parser.add_argument("-sim", "--sim-length-max", dest="sim_length_max", type=int, default=100,
                    help="Length of one successful run")

parser.add_argument("-m", "--memory-size", dest="replay_memory_size", type=int, default=5000,
                    help="Replay Memory Size")
parser.add_argument("-a", "--agents-number", dest="agents_number", type=int, default=2,
                    help="Number of agents to be used")

parser.add_argument("-lr", "--learning-rate", dest="learning_rate", type=float, default=2.5e-4,
                    help="NN learning rate")
parser.add_argument("-e", "--epochs-training", dest="n_epochs", type=int, default=40,
                    help="Number of epochs per training iteration")
parser.add_argument("-mb", "--minibatch-size", dest="minibatch_size", type=int, default=64,
                    help="Size of the minibatches used for training")

parser.add_argument("-y", "--epsilon", dest="epsilon", type=float, default=0.99,
                    help="initial epsilon")
parser.add_argument("-yd", "--epsilon-decay", dest="epsilon_decay", type=float, default=0.995,
                    help="annealing factor of epsilon")

parser.add_argument("-c", "--controller-probability", dest="ctrl_prob", type=float, default=0,
                    help="probability of control action being decided by conventional controller")
parser.add_argument("-cd", "--controller-probability-decay", dest="ctrl_prob_decay", type=float, default=0.9,
                    help="annealing factor of control probability")


parser.add_argument("-v", "--net-version", dest="net_version", type=int, default=0,
                    help="net version used")


args = parser.parse_args()

#####




num_cpus = psutil.cpu_count(logical=False)
n_agents = 2*num_cpus -2

def main(net_version = 0, n_iterations = 2, ray_parallelize = False, record_computing_time = False, difficulty = 0,\
         save_override = False, load_iteration = -1, agents_number = n_agents, learning_rate= 0.001,\
             n_epochs = 400, replay_memory_size = 5000, epsilon = .9, ctrlr_probability = 0, sim_length_max = 100, \
        epsilon_annealing_factor = 0.95,  ctrlr_prob_annealing_factor = 0.9 , mini_batch_size = 64, save_frequency = 1  ):

    if ray_parallelize:
        import ray

        # Start Ray.
        try:
            ray.shutdown()
        except Exception:
            print('ray not active')
        ray.init()

    #
    # to debug compiling time
    if record_computing_time:
        #dqn_env.plot_training_log()
        import cProfile
        import pstats
        import io
        pr = cProfile.Profile()
        pr.enable()
        
    # to debug compiling time
    #
    #dqn_env = Multiprocess_dqn_environment('RobotEnv', 'ConvModel', n_agents = num_cpus -2, ray_parallelize=ray_parallelize, move_to_cuda=True)
    
    #print(f'c = {ctrlr_probability}')
    #print(f'y = {epsilon}')
    
    # leave only simulation options here
    dqn_env = Multiprocess_DQN_Environment('CartPole', 'LinearModel', net_version , n_agents = agents_number, \
                                         ray_parallelize=ray_parallelize, move_to_cuda=True, n_frames = 1, \
                                         replay_memory_size = replay_memory_size, N_epochs = n_epochs, \
                                         save_override = save_override , epsilon = epsilon , \
                                         ctrlr_probability = ctrlr_probability, epsilon_annealing_factor=epsilon_annealing_factor,\
                                         ctrlr_prob_annealing_factor = ctrlr_prob_annealing_factor, discr_env_bins = 8 , \
                                         mini_batch_size = mini_batch_size, save_frequency=save_frequency, \
                                         difficulty = difficulty, learning_rate = learning_rate, sim_length_max = sim_length_max)

    # model load
    #dqn_env.load_model(3)
    #parameters override

    #dqn_env.movie_frequency = 2
    dqn_env.save_movie = False
    dqn_env.live_plot = False
    # always update agents params after dqn_env params are changed
    dqn_env.updateAgentsAttributesExcept('env')

    # launch env
    time_init = time.time()
        
    """
    # first run is to store memory
    print('first run started')
    dqn_env.runEnvIterationOnce()
    print('first run completed')    
    """
    
    # second run is to test parallel computation fo simulations and NN update
    if load_iteration != 0:
        dqn_env.load( load_iteration)
    dqn_env.runSequence(n_iterations) 
    
    #
    # to debug compiling time
    if record_computing_time:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        with open('test.txt', 'w+') as f:
            f.write(s.getvalue())
    # to debug compiling time
    #
    return dqn_env


if __name__ == "__main__":
    #"""
    env = main(net_version = args.net_version, n_iterations = args.n_iterations, ray_parallelize= args.ray_parallelize, \
               save_override =  args.save_override, load_iteration =args.load_iteration, \
                   replay_memory_size = args.replay_memory_size, agents_number = args.agents_number, \
                n_epochs = args.n_epochs, epsilon = args.epsilon, ctrlr_probability = args.ctrl_prob, \
                    epsilon_annealing_factor = args.epsilon_decay, ctrlr_prob_annealing_factor = args.ctrl_prob_decay, \
                        mini_batch_size = args.minibatch_size, save_frequency = args.save_frequency, learning_rate = args.learning_rate, \
                            sim_length_max = args.sim_length_max, difficulty = args.difficulty)
    """
    env = main(net_version = 1, load_iteration = 0,n_iterations = 10, ray_parallelize = False, record_computing_time = False, \
         save_override = False, n_epochs = 200, replay_memory_size = 1000, epsilon = .99, ctrlr_probability = 0, \
        epsilon_annealing_factor = 0.98,  mini_batch_size = 128 )
    """