# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:23:05 2020

@author: Enrico Regolin
"""

# Importing the libraries
# required if current directory is not found
import sys 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    
from pathlib import Path as createPath
#%%
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from torch.nn.utils import clip_grad_value_ , clip_grad_norm_
import asyncio
import time

from copy import deepcopy

#%%
# my libraries

from .memory import ReplayMemory
#from rl.utilities import check_WhileTrue_timeout

#%%
DEBUG = False

class SimulationAgent:
    
    ##################################################################################
    def __init__(self,sim_agent_id, env = None, rl_mode = 'DQL', model_qv= None, model_pg = None,model_v = None,\
                 n_frames = 1, net_name = 'no_name' ,epsilon = 0.9, ctrlr_probability = 0, save_movie = False, \
                 max_consecutive_env_fails = 3, max_steps_single_run = 200, show_rendering = True, 
                 flip_gradient_sign = False, use_reinforce = False, \
                 tot_iterations = 1000, live_plot = False, verbosity = 0 , noise_sd = 0.05, \
                 movie_frequency = 10, max_n_single_runs = 1000, save_sequences  = False, reward_history_span = 200):
                
        self.reward_history = []
        self.reward_history_span = reward_history_span
        
        self.prob_correction = 0.2 # probability of "correction" of "non sense random inputs" generated
        
        self.flip_gradient_sign = flip_gradient_sign
        self.use_reinforce = use_reinforce
        
        self.beta_PG = 1 
        self.gamma = 0.99

        self.rl_mode = rl_mode
        
        self.save_sequences  = save_sequences
        self.reset_full_sequences()
        
        # for testing only
        self.max_n_single_runs = max_n_single_runs

        self.is_running = False
        self.share_conv_layers = False
        
        self.sim_agent_id = sim_agent_id
        self.net_name = net_name
        
        self.save_movie = save_movie
        
        #self.move_to_cuda = torch.cuda.is_available() and move_to_cuda

        self.internal_memory_size = np.round(1.2*tot_iterations)
        self.internal_memory = ReplayMemory(size = self.internal_memory_size)
        
        # instantiate DQN (env is required for NN dimension)
        self.env = env
        
        if self.env is not None:
            self.n_actions = self.env.get_actions_structure()
        else:
            self.n_actions = None
        
        #(self.env.n_bins_act+1)**(self.env.act_shape[0])  # n_actions depends strictly on the type of environment
        
        if self.rl_mode != 'AC':
            self.model_qv = model_qv
        if 'AC' in self.rl_mode :
            self.model_pg = model_pg
            self.model_v = model_v
        
        self.n_frames = n_frames
        
        self.epsilon = epsilon
        self.ctrlr_probability = ctrlr_probability
        
        # verbosity determines how many messages are displayed
        self.tot_iterations = tot_iterations 
        self.verbosity = verbosity

        self.display_status_frequency = self.tot_iterations        
        if self.verbosity == 2:
            self.display_status_frequency = int(round(self.tot_iterations/10))
        elif self.verbosity == 1:
            self.display_status_frequency = int(round(self.tot_iterations/4))
        
        self.show_rendering = show_rendering
        self.live_plot = live_plot
        self.movie_frequency = movie_frequency
        
        self.max_consecutive_env_fails = max_consecutive_env_fails
        
        self.max_steps_single_run =  max_steps_single_run
        
    ########## required to avoid RL level setting max steps as None, while still allowing it to change it
    @property
    def max_steps_single_run(self):
        return self._max_steps_single_run
    @max_steps_single_run.setter
    def max_steps_single_run(self,value):
        if self.env is not None:
            self._max_steps_single_run = np.maximum(value, self.env.get_max_iterations())
        else:
            self._max_steps_single_run = value
        
        
    ##################################################################################        
    #initialize variables at the beginning of each run (to reset iterations) 
    def run_variables_init(self):
       
        self.loss_policy = 0 
       
        self.agent_run_variables = dict(
            iteration = 1,
            single_run = 0,
            cum_reward = 0,
            fails_count = 0,
            consecutive_fails = 0,
            steps_since_start = 0,
            failed_iteration = False,
            successful_runs = 0)

    """
    ##################################################################################        
    # required since ray wrapper doesn't allow accessing attributes
    def hasInternalMemoryInfo(self, threshold = 0):
        return self.internal_memory.fill_ratio >= threshold
    """

    ##################################################################################        
    # required since ray wrapper doesn't allow accessing attributes
    async def isRunning(self):
        return self.is_running

    ##################################################################################        
    # required to check if model is inherited
    def getAttributeValue(self, attribute):
        if attribute in self.__dict__.keys():
            return self.__dict__[attribute]  

    ##################################################################################        
    # required since ray wrapper doesn't allow accessing attributes
    def getAttributes(self):
        return [key for key in self.__dict__.keys()]  
        
    ##################################################################################        
    # required since ray wrapper doesn't allow accessing attributes
    def setAttribute(self,attribute,value):
        if attribute in self.__dict__.keys():
            self.__dict__[attribute] = value
            
            
    ##################################################################################        
    def update_model_weights_only(self, model_weights, model = 'model_qv'):
        """ updates model_pg weights based on external model leaving the optimizer in its current state"""
        for k,v in model_weights.items():
            self.__dict__[model].state_dict()[k] *= 0
            self.__dict__[model].state_dict()[k] += v
        
    ##################################################################################        
    def renderAnimation(self, action = 0, done = False, reward_np=0):
        iter_params = list( map(self.agent_run_variables.get, ['iteration', 'single_run', 'cum_reward']) )
        if self.env.env_type == 'RobotEnv':
            self.renderRobotEnv(iter_params, done, reward_np)
        elif self.env.env_type == 'CartPole':
            self.renderCartPole(iter_params, action)

    ##################################################################################        
    def renderCartPole(self, iter_params, action):
        self.env.render(iter_params=iter_params, action = action)
            
    ##################################################################################        
    def renderRobotEnv(self, iter_params, done, reward_np):
        # render animation
        
        if self.show_rendering:
            if self.live_plot:
                self.env.render('plot',iter_params=iter_params)
                plt.show()
            elif not (self.agent_run_variables['single_run'] % self.movie_frequency) :
                frame = self.env.render('animation',iter_params=iter_params)
                self.ims.append(frame) 
                if done:
                    iter_params[0]+=1
                    iter_params[2]+=reward_np
                    frame = self.env.render('animation',iter_params=iter_params)
                    for _ in range(10):
                        self.ims.append(frame) 
        
    ##################################################################################
    def controller_selector(self, pctg_ctrl = 0):
        if pctg_ctrl > 0:
            return random.random() <= pctg_ctrl
        else:
            return random.random() <= self.ctrlr_probability
    
    ##################################################################################
    def reset_full_sequences(self):
        self.states_full_sequences = []
        self.ctrl_full_sequences = []        
        
    ##################################################################################
    def get_full_sequences(self):
        return self.states_full_sequences , self.ctrl_full_sequences        
    
    ##################################################################################
    def get_full_sequence_and_reset(self, include_ctrl = False):
        states,ctrls = self.get_full_sequences()
        self.reset_full_sequences()
        if include_ctrl:
            return states,ctrls
        else:
            return states
        
    ##################################################################################
    def reset_agent(self, pctg_ctrl = 0, evaluate = False, info = None):
        
        if self.save_sequences and self.env.env.get_complete_sequence() is not None:
            state_sequence, ctrl_sequence = self.env.env.get_complete_sequence()
            if not hasattr(self, 'states_full_sequences'):
                self.states_full_sequences = [state_sequence]
                self.ctrl_full_sequences = [ctrl_sequence]
            else:
                self.states_full_sequences.append(state_sequence)
                self.ctrl_full_sequences.append(ctrl_sequence)
        
        state_obs = self.env.reset(save_history = (not self.agent_run_variables['single_run'] % self.movie_frequency), evaluate = evaluate)
        
        if state_obs is None:
            # initial action is do nothing
            action = torch.zeros([self.n_actions], dtype=torch.bool)
            action[round((self.n_actions-1)/2)] = 1
            state_obs, reward, done, info = self.env.action(action)
            
        state = self.env.get_net_input(state_obs, reset = True)
        
        single_run_log = self.trainVariablesUpdate(reset_variables = True, info = info)
        self.update_sim_log(single_run_log)
        self.reset_simulation = False
        
        # the decision to use the controller in a given run (instead of the model output) is taken before the run starts
        use_controller = self.controller_selector(pctg_ctrl)
        
        return state, use_controller

    ##################################################################################
    def update_sim_log(self, single_run_log):
        
        if single_run_log is not None:
            
            if single_run_log[0,0] > 1 and not np.isnan(single_run_log[0,1] ):
            
                if self.simulation_log is None:
                    self.simulation_log = single_run_log
                else:
                    self.simulation_log = np.append(self.simulation_log, single_run_log, axis = 0)
                
        
    ##################################################################################
    def initialize_pg_lists(self):
        self.traj_rewards = []
        self.traj_state_value = []
        self.traj_log_prob = []
        self.traj_entropy = []
        self.advantage_loss = 0
        self.loss_policy = 0
        self.map_est_loss = 0
        self.state_sequences = []
        
        self.map_est_sequences = []

    
    ##################################################################################
    def pg_calculation(self, reward, state_0, prob, action_idx, done, info):
        """ core of PG functionality. here gradients are calculated (model update occurs in rl_env)"""
        
        valid = (info['outcome'] != 'fail')
        
        if valid:
            loss_pg_i = 0
            loss_v_i = 0
            loss_map_i = 0
            
            advantage = torch.tensor(0)
            entropy = torch.sum(-torch.log(prob)*prob)
            
            self.traj_rewards.append(reward)
            
            if self.rl_mode == 'AC':
                with torch.no_grad():
                    self.traj_state_value.append(self.model_v(state_0))
            elif self.rl_mode == 'parallelAC':
                with torch.no_grad():
                    self.traj_state_value.append(torch.max(self.model_qv(state_0)))
                    
            self.traj_log_prob.append(torch.log(prob[:,action_idx]))
            self.traj_entropy.append(entropy)
            self.state_sequences.append(state_0) 
            
            if self.env.env_type == 'RobotEnv' and info['robot_map'] is not None and self.model_pg.partial_outputs:
                self.map_est_sequences.append(info['robot_map'])
                
            if done:
                R = 0
                for i in range(len(self.traj_rewards)):
                    R = self.traj_rewards[-1-i] + self.gamma* R
                    
                    if self.use_reinforce:
                        advantage = R
                    else:
                        advantage = R - self.traj_state_value[-1-i]

                    if self.flip_gradient_sign:
                        self.loss_policy += advantage*self.traj_log_prob[-1-i] - self.beta_PG*self.traj_entropy[-1-i]
                        if DEBUG:
                            print('sign changed')
                    else:
                        self.loss_policy += -advantage*self.traj_log_prob[-1-i] - self.beta_PG*self.traj_entropy[-1-i]
                        
                    #self.advantage_loss += (  R - torch.max(self.model_qv(self.state_sequences[-1-i].float())) )**2
                    if self.rl_mode == 'AC':
                        state_value, output_map = self.model_v(self.state_sequences[-1-i], return_map = True) #.float())
                        self.advantage_loss += (  R -  state_value )**2
                        
                        if self.env.env_type == 'RobotEnv' and info['robot_map'] is not None and output_map is not None:
                            self.map_est_loss += torch.sum( (torch.tensor(info['robot_map'])- output_map)**2 )
                            map_loss_scalar = self.map_est_loss.item()
                        else:
                            map_loss_scalar = 0
                       
                    #if not self.share_conv_layers:
                    #    total_loss = self.advantage_loss + self.loss_policy + self.map_est_loss
                    #else:
                    total_loss = self.advantage_loss/(1e-5+self.advantage_loss.item()) \
                                + self.loss_policy/(1e-5+abs(self.loss_policy.item())) \
                                + self.map_est_loss/(1e-5+map_loss_scalar)
                
                total_loss.backward()

                loss_pg_i = np.round(self.loss_policy.item(),3)
                if self.rl_mode == 'AC':
                    loss_v_i = np.round(self.advantage_loss.item(),3)
                    if torch.is_tensor(self.map_est_loss):
                        loss_map_i = np.round(self.map_est_loss.item(),3)
                                        
                self.initialize_pg_lists()

            return loss_pg_i, entropy.item(), loss_v_i, loss_map_i

        else:
            self.loss_policy = 0
            return np.nan, np.nan, np.nan, np.nan

    ##################################################################################
    def initialize_run(self):
        """ initialize sim run environment"""
        
        force_stop = False

        if 'AC' in self.rl_mode:
            # check if model contains nan
            for k,v in self.model_pg.state_dict().items():
                if torch.isnan(v).any():
                    print('nan tensor from start')
                    force_stop = True
            self.model_pg.optimizer.zero_grad()
            self.model_v.optimizer.zero_grad()
            self.initialize_pg_lists()

        # initialize environment
        fig_film = None
        if self.show_rendering:
            self.ims = []
            fig_film = plt.figure()
            if self.live_plot:
                self.env.render_mode = 'plot'
                
        self.reset_simulation = True
        self.stop_run = False
        self.simulation_log = None
        self.run_variables_init()

        self.pg_loss_hist = []
        self.entropy_hist = []
        self.advantage_hist = []
        self.loss_map_hist = []
        done = False
        
        return force_stop, done, fig_film


    ##################################################################################
    def sim_routine(self, state, loss_pg , use_controller, use_NN, test_qv):
        """ single iteration routine for env simulation"""
        
        force_stop = False
        
        action, action_index, noise_added, prob_distrib = self.getNextAction(state, use_controller=use_controller, use_NN = use_NN, test_qv=test_qv )
        
        if use_NN and DEBUG:
            print(f'action_index: {action_index}')
            
        state_1, reward_np, done , info = self.stepAndRecord(state, action, action_index, noise_added)
        
        if use_NN and (info['outcome'] is not None) and (info['outcome'] != 'fail') and (info['outcome'] != 'opponent'):
            self.agent_run_variables['successful_runs'] += 1
            
        if 'AC' in self.rl_mode:
            loss_pg_i, entropy_i, advantage_i, loss_map_i = self.pg_calculation(reward_np, state, prob_distrib, action_index, done, info )
            loss_pg += loss_pg_i
            force_stop = np.isnan(loss_pg_i)
            self.entropy_hist.append(entropy_i)
            self.advantage_hist.append(advantage_i)
            self.loss_map_hist.append(loss_map_i)
            if done:
                self.pg_loss_hist.append(loss_pg)
        
        if self.verbosity > 0:
            self.displayStatus()                    
        self.trainVariablesUpdate(reward_np, done, force_stop, no_step = ('no step' in info) )
        
        state = state_1

        return done, state, force_stop  , loss_pg, info


    ##################################################################################
    def extract_gradients(self, force_stop):
        """ extract gradients from PG and V model to be shared with main model for update"""
        
        pg_info = None
        nan_grad = False
        if 'AC' in self.rl_mode:
            if force_stop:
                pg_info = (None,None,None,None,None,None, False)
            else:
                grad_dict_pg = {}
                grad_dict_v = {}
                for name, param in self.model_pg.named_parameters():
                    if param.grad is not None:
                        grad_dict_pg[name] = param.grad.clone()
                    else:
                        grad_dict_pg[name] = 0
                        
                if self.rl_mode == 'AC':
                    if param.grad is not None:

                        for name, param in self.model_v.named_parameters():
                            if torch.isnan(param.grad).any():
                                nan_grad = True
                                break
                            else:
                                grad_dict_v[name] = param.grad.clone()
                                
                    else:
                        grad_dict_v[name] = 0


                if not nan_grad:
                    pg_info = (grad_dict_pg, grad_dict_v, np.average(self.pg_loss_hist) , \
                           np.average(self.entropy_hist),np.average(self.advantage_hist), \
                           np.average(self.loss_map_hist), True )
                else:
                    pg_info = (None,None,None,None,None,None,False)
                    print('remote worker has diverging gradients')
        return pg_info


    ##################################################################################
    def run_synch(self, use_NN = False, pctg_ctrl = 0, test_qv = False):
        """synchronous implementation of sim-run """
        
        force_stop, done, fig_film = self.initialize_run()
        info = {}
        
        while not (self.stop_run or ( 'AC' in self.rl_mode and done and not use_NN ) ):
                
            if self.reset_simulation:
                loss_pg = 0
                if self.agent_run_variables['single_run'] >= self.max_n_single_runs+1:
                    break
                state, use_controller = self.reset_agent(pctg_ctrl, evaluate = use_NN , info = info)
           
            done, state, force_stop  , loss_pg, info = self.sim_routine(state, loss_pg , use_controller, use_NN, test_qv)
            
            if DEBUG and use_NN and self.rl_mode == 'AC':
                state_value = self.model_v(state)
                print(f'state value: {state_value}')
                

        single_run_log = self.trainVariablesUpdate(reset_variables = True, info = info)
        self.update_sim_log(single_run_log)
        self.endOfRunRoutine(fig_film = fig_film)
        plt.close(fig_film)
        
        pg_info = self.extract_gradients(force_stop)

        return self.simulation_log, self.agent_run_variables['single_run'], self.agent_run_variables['successful_runs'], self.internal_memory.fill_ratio,  pg_info #[0]
        # self.simulation_log contains duration and cumulative reward of every single-run


    ##################################################################################
    async def run(self, use_NN = False, pctg_ctrl = 0, test_qv = False):
        """synchronous implementation of sim-run """
        
        force_stop, done, fig_film = self.initialize_run()
        info = {}
        self.is_running = True
        self.external_force_stop = False
        
        while not (self.stop_run or ( 'AC' in self.rl_mode and done) or self.external_force_stop):
            
            await asyncio.sleep(0.00001)
                
            if self.reset_simulation:
                loss_pg = 0
                if self.agent_run_variables['single_run'] >= self.max_n_single_runs+1:
                    break
                state, use_controller = self.reset_agent(pctg_ctrl, evaluate = use_NN )
           
            done, state, force_stop  , loss_pg, info = self.sim_routine(state, loss_pg , use_controller, use_NN, test_qv)

        single_run_log = self.trainVariablesUpdate(reset_variables = True, info = info)
        self.update_sim_log(single_run_log)
        self.endOfRunRoutine(fig_film = fig_film)
        plt.close(fig_film)
        
        pg_info = self.extract_gradients(self.external_force_stop or force_stop)
        self.is_running = False

        return self.simulation_log, self.agent_run_variables['single_run'], self.agent_run_variables['successful_runs'], self.internal_memory.fill_ratio,  pg_info #[0]
        # self.simulation_log contains duration and cumulative reward of every single-run


    ################################################################################
    async def force_stop(self):
        self.external_force_stop = True

        
    ################################################################################
    def emptyLocalMemory(self):
        return self.internal_memory.getMemoryAndEmpty()
        
        
    ################################################################################
    def stepAndRecord(self,state, action,action_index, noise_added):
        
        #################
        try:
            # in case of infeasibility issues, random_gen allows a feasible input to be re-generated inside the environment, to accelerate the learning process
            action_bool_array = action.detach().numpy()
            state_obs_1, reward_np, done, info = self.env.action(action_bool_array)
            
            if np.isnan(reward_np) or np.isinf(reward_np):
                self.agent_run_variables['failed_iteration'] = True
            
            elif 'move changed' in info:
                action = 0*action
                action_index = self.env.get_action_idx(info['move changed'])
                action[ action_index ] = 1
            elif self.show_rendering and DEBUG:
                # we show the selected action of the not corrected ones
                print(f'selected action = {self.env.boolarray_to_action(action_bool_array)}')

        except Exception:
            self.agent_run_variables['failed_iteration'] = True

        ################# 
        if not self.agent_run_variables['failed_iteration']:

            self.renderAnimation(action, done, reward_np)
            ## restructure new data
            state_1 = self.env.get_net_input(state_obs_1, state_tensor_z1 = state )

            action = action.unsqueeze(0)
            reward = torch.from_numpy(np.array([reward_np], dtype=np.float32)).unsqueeze(0)
            
            if self.rl_mode != 'AC':
                # build "new transition" and add it to replay memory
                new_transition = (state, action, reward, state_1, done, action_index)
                self.internal_memory.addInstance(new_transition)
            
            if not 'outcome' in info:
                info['outcome'] = None
                
        else:
            reward_np = np.nan
            state = None
            state_1 = None
            done = None
            info = {'outcome' : 'fail'}
            
        return state_1, reward_np, done, info
    
    ##################################################################################
    def endOfRunRoutine(self, fig_film):
        # end of training routine
        if self.verbosity > 0:
            print(f'completed runs: {self.agent_run_variables["single_run"]}')
            display_failed_its = round(100*self.agent_run_variables["fails_count"]/self.agent_run_variables["iteration"],1)
            print(f"failed iterations: {display_failed_its}%")
            display_failed_runs = round(100*self.agent_run_variables["fails_count"]/(self.agent_run_variables["fails_count"] + self.agent_run_variables["single_run"]),1)
            print(f"failed runs: {display_failed_runs}%")
        
        if self.show_rendering and self.agent_run_variables["single_run"]>=self.movie_frequency  and \
            not self.live_plot and not self.agent_run_variables["consecutive_fails"] >= 3:
            
            try:
                kernel_exec = False
                ip = get_ipython()
                if ip.has_trait('kernel'):
                    kernel_exec = True
            except Exception:
                kernel_exec = False

            if not kernel_exec:
                ani = None
                if self.env.env_type == 'RobotEnv':
                    ani , filename , duration=self.getVideoRobotEnv(fig_film)
                elif self.env.env_type == 'CartPole':
                    ani , filename, duration=self.getVideoCartPole(fig_film)
                
                if ani is not None:
                    #print(f'duration = {duration}s')
                    plt.show(block=False)
                    plt.waitforbuttonpress(round(duration))
                                
                    if self.save_movie:
                        if 'AC' in self.rl_mode :
                            net_type = self.model_pg.net_type
                        elif self.rl_mode == 'DQL':
                            net_type = self.model_qv.net_type
                        else:
                            raise('RL mode not defined')                        
                            
                        store_path= os.path.join( os.path.dirname(os.path.abspath(__file__)) ,"Data" , self.env.env_type, net_type, 'video'  )
                        createPath(store_path).mkdir(parents=True, exist_ok=True)
                        
                        full_filename = os.path.join(store_path, filename)
                        ani.save(full_filename)

    ##################################################################################
    def getVideoCartPole(self, fig_film):
        filename = self.net_name +'.mp4'
        interval = round(0.5*self.env.dt*1000)
        duration = round(0.001*len(self.ims)*interval,1)
        ani = self.env.get_gif(fig = fig_film, save_history = (not self.agent_run_variables['single_run'] % self.movie_frequency) )
        return ani, filename, duration

    ##################################################################################
    def getVideoRobotEnv(self, fig_film):
        # in case it was a live_plot test
        #if self.live_plot:
        #    self.env.render_mode = 'animation'
        filename = self.net_name +'_agent_id_'+ str(self.sim_agent_id) +'.mp4'
        interval = round(0.5*self.env.dt*1000)
        ani = animation.ArtistAnimation(fig_film, self.ims, interval=interval, blit=True)
        duration = round(0.001*len(self.ims)*interval,1)
        return ani, filename, duration

    ##################################################################################
    def getNextAction(self,state, use_controller = False, use_NN = False, test_qv = False):
        # initialize action
        prob_distrib = None
        noise_added = False
        action = torch.zeros([self.n_actions], dtype=torch.bool)
        # PG only uses greedy approach            
        if 'AC' in self.rl_mode:
            prob_distrib = self.model_pg.cpu()(state)
            #qvals = self.model_qv.cpu()(state.float())

            #if use_NN:
            #    action_index = torch.argmax(prob_distrib)
            #else:
            try:
                action_index = torch.multinomial(prob_distrib, 1, replacement=True)
            except Exception:
                action_index = torch.argmax(prob_distrib)
                print('torch.multinomial failed')
                
        elif self.rl_mode == 'DQL':
            if use_controller and not use_NN:
                action_index = torch.tensor(self.env.get_control_idx(discretized = True), dtype = torch.int8)
            elif use_NN:
                qvals = self.model_qv.cpu()(state)
                action_index = torch.argmax(qvals)
            else:
                if random.random() <= self.epsilon and not use_NN:
                    #action_index = torch.randint(self.n_actions, torch.Size([]), dtype=torch.int8)
                    #action_index = torch.randint(self.n_actions, (1,), dtype=torch.int8)
                    action_index = torch.tensor(random.randint(0,self.n_actions-1))
                else:
                    # this function allows to randomly choose other good performing q_values
                    qvals = self.model_qv.cpu()(state)
                    action_index = prob_action_idx_qval(qvals)     
            
        action[action_index] = 1
        
        return action, action_index, noise_added, (prob_distrib if 'AC' in self.rl_mode else None )

            
    ##################################################################################
    def trainVariablesUpdate(self, reward_np = 0, done = False, force_stop = False, \
                             reset_variables = False, no_step = False, info = None):
        # output is the model output given a certain state in
        
        if reset_variables:

            if self.agent_run_variables['single_run']>0: 
                cum_reward_data = self.agent_run_variables['cum_reward']
                if self.env.env_type == 'Frx' and self.agent_run_variables['steps_since_start']>0:
                    if info is not None:
                        cum_reward_data = info['return']
                        #print(f'single run return: {cum_reward_data}')
                    else:
                        cum_reward_data = 0
                single_run_log = np.array([self.agent_run_variables['steps_since_start'] , cum_reward_data ] )[np.newaxis,:]
                    
            else:
                single_run_log = None                

            self.agent_run_variables['steps_since_start'] = 0
            self.agent_run_variables['failed_iteration'] = False
            
            self.agent_run_variables['single_run'] +=1
            
            self.reward_history.append(self.agent_run_variables['cum_reward'])
            self.agent_run_variables['cum_reward'] = 0

            return single_run_log
            
        else:
            if self.agent_run_variables['failed_iteration']:
                self.agent_run_variables['iteration'] += 1
                self.agent_run_variables['fails_count'] += 1
                self.agent_run_variables['consecutive_fails'] += 1
            else:
                ## end of iteration - training parameters update (also used in testing)
                self.agent_run_variables['iteration'] += 1
                self.agent_run_variables['consecutive_fails'] = 0
                # keep track of the length of the current run and of the gained rewards
                self.agent_run_variables['steps_since_start'] += 1*int(not no_step)
                self.agent_run_variables['cum_reward'] += reward_np[0] if isinstance(reward_np, np.ndarray) else reward_np 
            
            # update of reset_simulation and stop_run
            if self.agent_run_variables['failed_iteration'] or done or self.agent_run_variables['steps_since_start'] > self.max_steps_single_run +1 :  # (+1 is added to be able to verify the condition in the next step)
                self.reset_simulation = True

                if self.agent_run_variables['consecutive_fails'] >= self.max_consecutive_env_fails or self.agent_run_variables['iteration'] >= self.tot_iterations or force_stop:
                    self.stop_run = True
        pass
                    
    ##################################################################################
    def displayStatus(self, loss_np = []):
        if self.agent_run_variables['iteration'] % self.display_status_frequency == 0:  #update weights has not occurred but we're in the "display iteration"
            perc_completed = round(self.agent_run_variables['iteration']/self.tot_iterations*100,1)    
            if not loss_np:
                print(f'agent = {self.sim_agent_id}, iteration = {self.agent_run_variables["iteration"]}, completed = {round(perc_completed,1)}%')
            else:
                print(f"agent = {self.sim_agent_id}, iteration = {self.agent_run_variables['iteration']}, completed = {round(perc_completed,1)}%, loss = {np.round(loss_np,2)}, epsilon = {round(self.epsilon,2)}") 


def prob_action_idx_qval(qvals):
    _, indices = torch.sort(-qvals)
    p = np.zeros(indices.shape[1])
    for c,i in enumerate(indices.squeeze(0).numpy()):
        p[i] = 2.**(-c-1)
    return torch.multinomial(torch.tensor(p), 1, replacement=True).squeeze(0)

#%%
"""
from envs.gymstyle_envs import discr_GymStyle_Platoon
from nns.custom_networks import LinearModel

if __name__ == "__main__":
    #env = SimulationAgent(0, env = GymStyle_Robot(n_bins_act=2), model = ConvModel())
    pla_env = GymStyle_Platoon(n_bins_act=[10,1])
    
    agent = SimulationAgent(0, env=pla_env, n_frames=1  , model = \
                LinearModel('LinearModel_test', 0.001, pla_env.get_actions_structure(), \
                            pla_env.get_observations_structure(), 20, 50, 50, 20) ) 
    agent.max_steps_single_run = 200
    
    
    agent.save_movie = False
    agent.movie_frequency = 1000
    agent.tot_iterations = 200
    agent.run_synch(pctg_ctrl=1)
    agent.env.env.plot_graphs()
"""

#%%


from ...envs.gymstyle_envs import DiscrGymStyleRobot
from ...nns.robot_net import ConvModel

if __name__ == "__main__":
    #env = SimulationAgent(0, env = GymStyle_Robot(n_bins_act=2), model = ConvModel())
    gym_env = DiscrGymStyleRobot(n_bins_act=2)
    
    agent = SimulationAgent(0, env=gym_env, n_frames=4  , model_qv = ConvModel() )
    agent.max_steps_single_run = 50
    
    
    agent.save_movie = False
    agent.movie_frequency = 1
    agent.tot_iterations = 100
    agent.run_synch()


#%%

"""
from envs.gymstyle_envs import GymStyle_CartPole
from nns.custom_networks import LinearModel


if __name__ == "__main__":
    #env = SimulationAgent(0, env = GymStyle_Robot(n_bins_act=4), model = ConvModel())
    gym_env = GymStyle_CartPole(n_bins_act=40)
    
    agent = SimulationAgent(0, env=gym_env  , model = LinearModel('LinearModel_test', gym_env.get_actions_structure(), gym_env.get_observations_structure(), 10, 10) )
    agent.max_steps_single_run = 2000
    
    agent.movie_frequency = 1
    agent.tot_iterations = 1000
    agent.run(use_controller = True)
    agent.env.env.cartpole.plot_graphs(dt=agent.env.env.dt, save = False, no_norm = False, ml_ctrl = True)

"""