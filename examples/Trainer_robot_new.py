#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 17:08:05 2020

@author: Enrico Regolin
"""

#%%

# for time debug only
import cProfile
import pstats
import io
# for time debug only
import os
import yaml

from asynch_rl.rl.rl_env import Multiprocess_RL_Environment
from asynch_rl.rl.utilities import clear_pycache, store_train_params, load_train_params

import psutil
import time
import ray





num_cpus = psutil.cpu_count(logical=False)
n_agents = 2*num_cpus -2

def run_training(configs):
    function_inputs = locals().copy()
    print(function_inputs)

    env_type = 'RobotEnv'
    model_type = 'ConvModel'
    overwrite_params = ['rewards', 'rl_mode', 'share_conv_layers', 'n_frames' ,\
                        'layers_width', 'map_output', 'normalize_layers', 'agents_number',\
                            'val_frequency']


    # initialize required net and model parameters if loading from saved values
    if configs['load_iteration'] != 0:

        storage_path = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) ,"Data" , \
                                env_type, model_type+str(configs['net_version']) )

        if os.path.isfile(os.path.join(storage_path,'train_params.txt')):
            my_dict = load_train_params(env_type, model_type, overwrite_params, configs['net_version'])
            for i,par in enumerate(overwrite_params):
                exec(par + " =  my_dict['"  + par + "']")
            del( overwrite_params, my_dict)

    # import ray
    if configs['ray_parallelize']:
        # Start Ray.

        configs['replay_memory_size'] *= configs['agents_number']
        #replay_memory_size *= 25 # (to ensure same epoch length between DQL on cluster and AC on eracle )

        try:
            ray.shutdown()
        except Exception:
            print('ray not active')

        if configs['ray_password'] is not None:
            ray.init(address=configs['head_address'], redis_password=configs['ray_password'] )
        else:
            ray.init()


    # launch env
    time_init = time.time()

    pr = cProfile.Profile()
    pr.enable()


    # second run is to test parallel computation fo simulations and NN update

    if configs['load_iteration'] > 0:
        print('###################################################################')
        print(f'Loading Environment. iteration: {configs["load_iteration"]}')
        print('###################################################################')

    rl_env = Multiprocess_RL_Environment(env_type , model_type , configs['net_version'] , rl_mode = configs['rl_mode'], \
                        ray_parallelize=configs['ray_parallelize'], move_to_cuda=True, n_frames = configs['n_frames'], \
                        replay_memory_size = configs['replay_memory_size'], n_agents = configs['agents_number'],\
                        tot_iterations = configs['tot_iterations'], discr_env_bins = 2 , \
                        use_reinforce = configs['use_reinforce'],  epsilon_annealing_factor=configs['epsilon_decay'],  layers_width= configs['layers_list'],\
                        N_epochs = configs['n_epochs'], epsilon_min = configs['epsilon_min'] , rewards = configs['rewards_list'], \
                        mini_batch_size = configs['minibatch_size'], share_conv_layers = configs['share_conv_layers'], \
                        difficulty = configs['difficulty'], learning_rate = configs['learning_rate'], sim_length_max = configs['sim_length_max'], \
                        memory_turnover_ratio = configs['memory_turnover_ratio'], val_frequency = configs['val_frequency'] ,\
                        gamma = configs['gamma'], beta_PG = configs['beta'] , continuous_qv_update = configs['continuous_qv_update'] , \
                        memory_save_load = configs['memory_save_load'] , normalize_layers = configs['normalize_layers'], \
                            map_output = configs['map_output'], peds_speed_mult = configs['peds_speed_mult'], downsampling_step = configs['downsampling_step'])

    # always update agents params after rl_env params are changed
    rl_env.updateAgentsAttributesExcept('env')


    if configs['load_iteration'] != 0:
        rl_env.load( configs['load_iteration'])

    else:
        store_train_params(rl_env, function_inputs)

    rl_env.runSequence(configs['n_iterations'], reset_optimizer=configs['reset_optimizer'], update = not configs['tester'])

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()
    with open('duration_test.txt', 'w+') as f:
        f.write(s.getvalue())

    return rl_env



if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    with open("trainer_robot_config.yaml", 'r') as f:
        configs = yaml.safe_load(f)
    env = run_training(configs)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()
    with open('duration_trainer_robot.txt', 'w+') as f:
        f.write(s.getvalue())
    current_folder = os.path.abspath(os.path.dirname(__file__))
    clear_pycache(current_folder)