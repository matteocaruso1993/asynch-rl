#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:16:15 2020

@author: rodpod21
"""

# tester cartpole


from DQN_env import Multiprocess_DQN_Environment

import sys
import psutil
import time



# df loader

dqn_env = Multiprocess_DQN_Environment('CartPole', 'LinearModel', 20, ray_parallelize=False, difficulty=1, \
                                     move_to_cuda=False, n_frames = 1, discr_env_bins=10, save_override= False, test_mode = True ) #, \
                                      #replay_memory_size = 500, N_epochs = 100)


dqn_env.save_movie = False
dqn_env.live_plot = False
# always update agents params after dqn_env params are changed
dqn_env.updateAgentsAttributesExcept('env')

dqn_env.load(-1)
#dqn_env.load(320)

dqn_env.plot_training_log()

#"""

agent = dqn_env.sim_agents[0]

#agent.max_steps_single_run = 20000

agent.movie_frequency = 1
#agent.tot_iterations = 10000
agent.max_n_single_runs = 1
agent.run()
agent.env.env.cartpole.plot_graphs(dt=agent.env.env.dt, save = False, no_norm = False, ml_ctrl = True)
#"""