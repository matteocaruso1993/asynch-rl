# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:23:05 2020

@author: enric
"""

# Importing the libraries
# required if current directory is not found
import sys 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

csfp = os.path.abspath(os.path.dirname(__file__))
if csfp not in sys.path:
    sys.path.insert(0, csfp)
    
from pathlib import Path as createPath
#%%
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import asyncio

#%%
# my libraries
from .memory import ReplayMemory

#%%

DEBUG = False

class SimulationAgent:
    
    ##################################################################################
    def __init__(self,sim_agent_id, env = None, rl_mode = 'DQL', model_qv= None, model_pg = None, n_frames = 1, \
                 net_name = 'no_name' ,epsilon = 0.9, ctrlr_probability = 0, \
                 internal_memory_size = 100, save_movie = True, \
                 max_consecutive_env_fails = 3, max_steps_single_run = 200, show_rendering = True, \
                 tot_iterations = 100, live_plot = False, verbosity = 0 , noise_sd = 0.05, \
                 movie_frequency = 10, max_n_single_runs = 1000, save_sequences  = False):
        
        self.noise_sd = noise_sd
        self.prob_correction = 0.2 # probability of "correction" of "non sense random inputs" generated

        self.rl_mode = rl_mode
        
        self.save_sequences  = save_sequences
        self.reset_full_sequences()
        
        # for testing only
        self.max_n_single_runs = max_n_single_runs

        self.is_running = False
        
        self.sim_agent_id = sim_agent_id
        self.net_name = net_name
        
        self.save_movie = save_movie
        
        #self.move_to_cuda = torch.cuda.is_available() and move_to_cuda

        self.internal_memory_size = internal_memory_size
        self.internal_memory = ReplayMemory(size = self.internal_memory_size)
        
        # instantiate DQN (env is required for NN dimension)
        self.env = env
        
        self.n_actions = self.env.get_actions_structure()
        #(self.env.n_bins_act+1)**(self.env.act_shape[0])  # n_actions depends strictly on the type of environment
        
        self.model_qv = model_qv
        if self.rl_mode == 'AC':
            self.model_pg = model_pg
        
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
        self._max_steps_single_run = np.maximum(value, self.env.get_max_iterations())
        
    ##################################################################################        
    #initialize variables at the beginning of each run (to reset iterations) 
    def run_variables_init(self):
       
        self.agent_run_variables = dict(
            iteration = 1,
            single_run = 0,
            cum_reward = 0,
            fails_count = 0,
            consecutive_fails = 0,
            steps_since_start = 0,
            failed_iteration = False,
            successful_runs = 0)

    ##################################################################################        
    # required since ray wrapper doesn't allow accessing attributes
    def hasInternalMemoryInfo(self, threshold = 0):
        return self.internal_memory.fill_ratio >= threshold

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
    def renderAnimation(self, action = 0):
        iter_params = list( map(self.agent_run_variables.get, ['iteration', 'single_run', 'cum_reward']) )
        if self.env.env_type == 'RobotEnv':
            self.renderRobotEnv(iter_params)
        elif self.env.env_type == 'CartPole':
            self.renderCartPole(iter_params, action)

    ##################################################################################        
    def renderCartPole(self, iter_params, action):
        self.env.render(iter_params=iter_params, action = action)
            
    ##################################################################################        
    def renderRobotEnv(self, iter_params):
        # render animation
        
        if self.show_rendering:
            if self.live_plot:
                self.env.render('plot',iter_params=iter_params)
                plt.show()
            elif not (self.agent_run_variables['single_run'] % self.movie_frequency) :
                self.ims.append(self.env.render('animation',iter_params=iter_params))        
        
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
    def reset_agent(self, pctg_ctrl = 0):
        
        if self.save_sequences and self.env.env.get_complete_sequence() is not None:
            state_sequence, ctrl_sequence = self.env.env.get_complete_sequence()
            if not hasattr(self, 'states_full_sequences'):
                self.states_full_sequences = [state_sequence]
                self.ctrl_full_sequences = [ctrl_sequence]
            else:
                self.states_full_sequences.append(state_sequence)
                self.ctrl_full_sequences.append(ctrl_sequence)
        
        self.env.reset(save_history = (not self.agent_run_variables['single_run'] % self.movie_frequency) )
        
        # initial action is do nothing
        action = torch.zeros([self.n_actions], dtype=torch.bool)
        action[round((self.n_actions-1)/2)] = 1
        
        state_obs, reward, done, info = self.env.action(action)
        state_obs_tensor = torch.from_numpy(state_obs).unsqueeze(0)  #added here
        
        # n_channels consecutive "images" of the state are used by the NN to predict --> we build the channels here (second to last dimension)
        # make it dependant on n_frames
        #state = torch.cat((state_obs_tensor, state_obs_tensor, state_obs_tensor, state_obs_tensor))  #.unsqueeze(0)
        if self.n_frames >1:
            state = state_obs_tensor.repeat(self.n_frames,1).unsqueeze(0)
        else:
            state = state_obs_tensor
        
        single_run_log = self.trainVariablesUpdate(reset_variables = True)
        self.update_sim_log(single_run_log)
        self.reset_simulation = False
        
        # the decision to use the controller in a given run (instead of the model output) is taken before the run starts
        use_controller = self.controller_selector(pctg_ctrl)
        
        return state, use_controller

    ##################################################################################
    def update_sim_log(self, single_run_log):
        
        if single_run_log is not None:
            if self.simulation_log is None:
                self.simulation_log = single_run_log
            else:
                self.simulation_log = np.append(self.simulation_log, single_run_log, axis = 0)
        
    ##################################################################################
    def run_synch(self, use_NN = False, pctg_ctrl = 0):
        
        self.simulation_log = None
        self.run_variables_init()
        
        # initialize environment
        fig_film = None
        if self.show_rendering:
            self.ims = []
            fig_film = plt.figure()
            if self.live_plot:
                self.env.render_mode = 'plot'
                
        self.reset_simulation = True
        self.stop_run = False
        
        while not self.stop_run:

            if self.reset_simulation:
                if self.agent_run_variables['single_run'] >= self.max_n_single_runs+1:
                    break
                state, use_controller = self.reset_agent(pctg_ctrl)
            
            action, action_index, noise_added = self.getNextAction(state, use_controller=use_controller, use_NN = use_NN)
            state, reward_np, done , info = self.stepAndRecord(state, action, action_index, noise_added)
            
            if use_NN and len(info)>0:
                self.agent_run_variables['successful_runs'] += 1
            
            if self.verbosity > 0:
                self.displayStatus()                    
            self.trainVariablesUpdate(reward_np, done)
            
        
        single_run_log = self.trainVariablesUpdate(reset_variables = True)
        self.update_sim_log(single_run_log)

        self.endOfRunRoutine(fig_film = fig_film)
        plt.close(fig_film)
        
        return self.simulation_log, self.agent_run_variables['single_run'], self.agent_run_variables['successful_runs'] #[0]
        # self.simulation_log contains duration and cumulative reward of every single-run

    ##################################################################################
    async def run(self, use_NN = False):
        
        self.simulation_log = None
        self.run_variables_init()
        
        self.is_running = True
        # initialize environment
        fig_film = None
        if self.show_rendering:
            self.ims = []
            fig_film = plt.figure()
            if self.live_plot:
                self.env.render_mode = 'plot'
                
        self.reset_simulation = True
        self.stop_run = False
        successful_runs = 0
        
        while not self.stop_run:
            
            await asyncio.sleep(0.00001)

            if self.reset_simulation:
                if self.agent_run_variables['single_run'] >= self.max_n_single_runs+1:
                    break
                state, use_controller = self.reset_agent()
            
            action, action_index, noise_added = self.getNextAction(state, use_controller=use_controller, use_NN = use_NN)
            state, reward_np, done , info = self.stepAndRecord(state, action, action_index, noise_added)
            
            if use_NN and len(info)>0:
                successful_runs += 1
            
            if self.verbosity > 0:
                self.displayStatus()                    
            self.trainVariablesUpdate(reward_np, done)
        
        single_run_log = self.trainVariablesUpdate(reset_variables = True)
        self.update_sim_log(single_run_log)

        self.endOfRunRoutine(fig_film = fig_film)
        self.is_running = False
        plt.close(fig_film)
        
        return self.simulation_log, self.agent_run_variables['single_run'], successful_runs #[0]
        # self.simulation_log contains duration and cumulative reward of every single-run
        
    ################################################################################
    def emptyLocalMemory(self, capacity_threshold = 0.5):
        if self.internal_memory.fill_ratio > capacity_threshold:
            return self.internal_memory.getMemoryAndEmpty()
        else:
            return []
        
    ################################################################################
    def stepAndRecord(self,state, action,action_index, noise_added):
        
        #################
        try:
            # in case of infeasibility issues, random_gen allows a feasible input to be re-generated inside the environment, to accelerate the learning process
            action_bool_array = action.detach().numpy()
            random_gen = (np.random.random() < self.prob_correction and noise_added )
            state_obs_1, reward_np, done, info = self.env.action(action_bool_array, random_gen = random_gen )
            if 'move changed' in info:
                action = 0*action
                action_index = self.env.get_action_idx(info['move changed'])
                action[ action_index ] = 1
            elif self.show_rendering or DEBUG:
                # we show the selected action of the not corrected ones
                print(f'selected action = {self.env.boolarray_to_action(action_bool_array)}')

        except Exception:
            self.agent_run_variables['failed_iteration'] = True
            reward_np = np.nan
            state = None
            state_1 = None
            done = None
            info = {}
        ################# 
        if not self.agent_run_variables['failed_iteration']:

            self.renderAnimation(action)
            ## restructure new data
            state_obs_1_tensor = torch.from_numpy(state_obs_1)
            
            if self.n_frames > 1:
                state_1 = torch.cat((state.squeeze(0)[1:, :], state_obs_1_tensor.unsqueeze(0))).unsqueeze(0)
            else:
                state_1 = state_obs_1_tensor.unsqueeze(0)

            action = action.unsqueeze(0)
            reward = torch.from_numpy(np.array([reward_np], dtype=np.float32)).unsqueeze(0)
            # build "new transition" and add it to replay memory
            new_transition = (state, action, reward, state_1, done, action_index)
            
            self.internal_memory.addInstance(new_transition)
            # set state to be state_1 (for next iteration)
            #state = state_1
            
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
        
        if self.agent_run_variables["single_run"]>=self.movie_frequency  and not self.live_plot and not self.agent_run_variables["consecutive_fails"] >= 3:
            
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
                    if self.rl_mode == 'AC':
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
        duration = None
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
    def getNextAction(self,state, use_controller = False, use_NN = False):
        # initialize action
        noise_added = False
        action = torch.zeros([self.n_actions], dtype=torch.bool)
        # PG only uses greedy approach            
        if self.rl_mode == 'AC':
            output = self.model_pg.cpu()(state.float())
            
            if random.random() <= self.epsilon and not use_NN:
                action_index = torch.argmax(output + self.noise_sd*torch.randn(output.shape))
                noise_added = True
            else:
                action_index = torch.argmax(output)

        elif self.rl_mode == 'DQL':
            if use_controller and not use_NN:
                action_index = torch.tensor(self.env.get_control_idx(discretized = True), dtype = torch.int8)
                #source = 'ctrl'
            else:
                if random.random() <= self.epsilon and not use_NN:
                    action_index = torch.randint(self.n_actions, torch.Size([]), dtype=torch.int8)
                    #source = 'random'
                else:
                    output = self.model_qv.cpu()(state.float())
                    action_index = torch.argmax(output)
                    #source = 'qv'
            
        action[action_index] = 1
        
        return action, action_index, noise_added
            
    ##################################################################################
    def trainVariablesUpdate(self, reward_np = 0, done = False, reset_variables = False):
        # output is the model output given a certain state in
        
        if reset_variables:

            if self.agent_run_variables['single_run']>0: 
                single_run_log = np.array([self.agent_run_variables['steps_since_start'] , self.agent_run_variables['cum_reward'] ] )[np.newaxis,:]
            else:
                single_run_log = None                

            self.agent_run_variables['steps_since_start'] = 0
            self.agent_run_variables['failed_iteration'] = False
            
            self.agent_run_variables['single_run'] +=1
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
                self.agent_run_variables['steps_since_start'] += 1
                self.agent_run_variables['cum_reward'] += reward_np[0] if isinstance(reward_np, np.ndarray) else reward_np 
            
            # update of reset_simulation and stop_run
            if self.agent_run_variables['failed_iteration'] or done or self.agent_run_variables['steps_since_start'] > self.max_steps_single_run +1 :  # (+1 is added to be able to verify the condition in the next step)
                self.reset_simulation = True
                                
                """
                print(f'failed iteration: {self.agent_run_variables["failed_iteration"]}')
                print(f'done: {done}')
                print(f'too many steps: {self.agent_run_variables["steps_since_start"] > self.max_steps_single_run +1}')
                """
                if self.agent_run_variables['consecutive_fails'] >= self.max_consecutive_env_fails or self.agent_run_variables['iteration'] >= self.tot_iterations:
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


#%%

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
    

#%%
"""
from envs.gymstyle_envs import GymStyle_Robot
from nns.custom_networks import ConvModel

if __name__ == "__main__":
    #env = SimulationAgent(0, env = GymStyle_Robot(n_bins_act=2), model = ConvModel())
    gym_env = GymStyle_Robot(n_bins_act=2)
    
    agent = SimulationAgent(0, env=gym_env, n_frames=4  , model = ConvModel() )
    agent.max_steps_single_run = 50
    
    
    agent.save_movie = False
    agent.movie_frequency = 1
    agent.tot_iterations = 100
    agent.run_synch()
"""

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