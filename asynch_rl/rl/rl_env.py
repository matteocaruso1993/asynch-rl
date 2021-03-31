# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:23:05 2020

@author: Enrico Regolin
"""

#%%
# System libraries
import sys 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%%
# External Libraries
import time
import ray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from bashplotlib.scatterplot import plot_scatter
from pathlib import Path as createPath

from copy import deepcopy
#%%
# My Libraries
from ..envs.gymstyle_envs import DiscrGymStyleRobot, DiscrGymStyleCartPole, \
    DiscrGymStylePlatoon, GymstyleChess, GymstyleConnect4, GymstyleDicePoker, GymstyleFrx
from ..nns.custom_networks import LinearModel
from ..nns.robot_net import ConvModel
from ..nns.nn_frx import NN_frx

from .resources.memory import ReplayMemory
from .resources.sim_agent import SimulationAgent
from .resources.updater import RL_Updater 
from .resources.parallelize import Ray_RL_Updater, Ray_SimulationAgent

from .utilities import progress, lineno, check_WhileTrue_timeout

#%%

DEBUG_MODEL_VERSIONS = True


#%%
class Multiprocess_RL_Environment:
    def __init__(self, env_type , net_type , net_version ,rl_mode = 'DQL', n_agents = 1, ray_parallelize = False , \
                 move_to_cuda = True, show_rendering = False , n_frames=4, learning_rate = [0.0001, 0.001] , \
                 movie_frequency = 20, max_steps_single_run = 200, \
                 training_session_number = None, save_movie = True, live_plot = False, \
                 display_status_frequency = 50, max_consecutive_env_fails = 3, tot_iterations = 400,\
                 epsilon = 1, epsilon_annealing_factor = 0.95,  \
                 epsilon_min = 0.01 , gamma = 0.99 , discr_env_bins = 10 , N_epochs = 1000, \
                 replay_memory_size = 10000, mini_batch_size = 512, sim_length_max = 100, \
                 ctrlr_prob_annealing_factor = .9,  ctrlr_probability = 0, difficulty = 0, \
                 memory_turnover_ratio = 0.2, val_frequency = 10, bashplot = False, layers_width= (5,5), \
                 rewards = np.ones(5), env_options = {}, share_conv_layers = False, 
                 beta_PG = 1 , continuous_qv_update = False, memory_save_load = False, \
                 flip_grad_sign = False, use_reinforce = False):
        
        
        self.flip_grad_sign = flip_grad_sign
        self.use_reinforce = use_reinforce
        
        self.continuous_qv_update = continuous_qv_update
        self.one_vs_one = False
        self.unstable_model = [False, False] # current model [0] and previous iteration [1] instability check. if occurrs in consecutive tries, update is aborted
        
        self.memory_save_load = memory_save_load
        
        ####################### net/environment initialization
        # check if env/net types make sense
        allowed_envs = ['RobotEnv', 'CartPole', 'Platoon', 'Chess', 'Connect4', 'DicePoker','Frx']
        allowed_nets = ['ConvModel', 'LinearModel','ConvFrxModel'] 
        allowed_rl_modes = ['DQL', 'AC', 'parallelAC']
        
        self.net_version = net_version
        
        if net_type not in allowed_nets or env_type not in allowed_envs or rl_mode not in allowed_rl_modes:
            raise('Env/Net Type not known')
            
        # net type has to be declared at every instanciation. net_type + ts_number define the net_name        
        self.net_type = net_type
        self.env_type = env_type
        self.rl_mode = rl_mode
        
         
        if 'AC' in self.rl_mode:
            self.global_pg_loss = []
            self.global_pg_entropy = []
            self.global_tr_sess_num = []
            self.global_cum_reward = []
            self.global_duration = []
            self.global_advantage = []
        
        self.training_session_number = 0 #by default ==0, if saved net is loaded this will be changed
        self.update_net_name()
        
        #######################
        # training options
        self.bashplot = bashplot 
        self.difficulty = difficulty
        self.sim_length_max = sim_length_max # seconds after which simulation is over (and successful)
        if np.isscalar(learning_rate):
            self.lr = [learning_rate, learning_rate]
        else:
            self.lr = learning_rate
            
        self.val_frequency = val_frequency
        self.ray_parallelize = ray_parallelize
        self.resume_epsilon = False
        self.share_conv_layers = share_conv_layers
        
        ####################### storage options        
        # storage path changes for every model
        self.storage_path = os.path.join( os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) ,"Data" , self.env_type, self.net_type+str(self.net_version) )
        createPath(self.storage_path).mkdir(parents=True, exist_ok=True)

        #######################
        # training parameters
        self.gamma = gamma
        self.n_epochs = N_epochs
        self.mini_batch_size = mini_batch_size
        self.memory_turnover_ratio = memory_turnover_ratio

        self.replay_memory_size = replay_memory_size
        self.memory_sizes_updater()
        
        self.beta_PG = beta_PG
        
        ####################### required to set up sim_agents agents created at the end)
        # NN options
        self.move_to_cuda = torch.cuda.is_available() and move_to_cuda and self.rl_mode != 'AC'
        print(f'using CUDA: {self.move_to_cuda}')
        
        if self.net_type in ['ConvModel', 'ConvFrxModel'] : # if we don't use conv_net there is no reason to use multichannel structure
            self.n_frames = n_frames # for multichannel convolutional networks only
        else:
            self.n_frames = 1
        self.layers_width = layers_width
        
        #######################
        # sim env options
        self.use_continuous_act_env = False
        self.discr_env_bins = discr_env_bins
        
        self.sim_agents_discr = []
        self.sim_agents_contn = []

        # agents number depends...
        self.n_agents_discr = 1
        if self.ray_parallelize:
            if self.use_continuous_act_env:
                self.n_agents_discr = 0
                self.n_agents_contn = n_agents
            else:
                self.n_agents_discr = n_agents
                self.n_agents_contn = 0                    

        self.rewards = rewards
        self.env_options = env_options

        ####################### run options
        #run duration
        self.tot_iterations = tot_iterations #(mìnimum) number of iterations of each  individual simulation run
        # end of run/end of session parameters
        self.max_steps_single_run = max_steps_single_run
        self.max_consecutive_env_fails = max_consecutive_env_fails

        #######################
        # epsilon/ctrl parameters
        self.epsilon = epsilon  # works also as initial epsilon in case of annealing
        self.epsilon_annealing_factor = epsilon_annealing_factor #○ epsilon reduction after every epoch [0<fact<=1]
        self.epsilon_min = epsilon_min
        
        self.ctrlr_probability = ctrlr_probability
        self.ctrlr_prob_annealing_factor = ctrlr_prob_annealing_factor

        #######################
        # visual interface options
        self.show_rendering = show_rendering
        self.movie_frequency = movie_frequency
        self.save_movie = save_movie        
        self.live_plot = live_plot   #only for debug purposes -> overrides save_movie
        self.display_status_frequency = display_status_frequency

        #######################################
        # models init
        self.model_qv = None
        self.model_pg = None
        self.model_v = None
        
        # added after configuration save (these attributes can't be saved in JSON format)
        self.env_discr = self.generateDiscrActSpace_GymStyleEnv()
               
        # this env can be used as reference and to get topologic information, but is not used for simulations!
        #(those are inside each agent)
        self.n_actions_discr = self.env_discr.get_actions_structure()
        prob_even = 1/self.n_actions_discr*torch.ones(self.n_actions_discr)
        self.max_entropy = np.round(torch.sum(-torch.log(prob_even)*prob_even).item(),3)
        print(f'max entropy = {self.max_entropy}')
                
        self.N_in_model = self.env_discr.get_observations_structure()

        if self.rl_mode != 'AC':
            self.model_qv = self.generate_model('qv')
        if self.rl_mode != 'DQL':
            self.model_pg = self.generate_model('pg')
            self.model_v = self.generate_model('v')
        
        # in one vs. one the environment requires the (latest) model to work
        if self.one_vs_one:
            self.env_discr.update_model(self.model_qv)
        
        # this part is basically only needed for 1vs1 (environment has to be updated within sim-agents)
        self.env = self.env_discr

        #######################
        # Replay Memory init
        self.shared_memory = ReplayMemory(size = self.replay_memory_size, minibatch_size= self.mini_batch_size)
        
        #######################
        # empty initializations
        self.qval_loss_history = self.qval_loss_hist_iteration = self.val_history = None
        self.nn_updater = None
        self.memory_stored = None
        
                # update log and history
        # (log structure: ['training session','epsilon', 'total single-run(s)','average single-run duration',\
        #    'average single-run reward' , 'average qval loss','average policy loss', 'N epochs', 'memory size', \
        #    action gen splits (0,1,2)..., 'pg_entropy']  )

        self.log_df = pd.DataFrame(columns=['training session','epsilon', 'total single-run(s)',\
                                    'average single-run duration', 'average single-run reward' , 'av. q-val loss', \
                                    'av. policy loss','N epochs', 'memory size', 'split random/noisy','split conventional',\
                                        'split NN', 'pg_entropy'])
            
        self.session_log = {'total runs':None, 'steps since start': None, \
                            'cumulative reward':None, 'qval_loss':None, \
                                'policy_loss':None, 'pg_entropy' : None}
            
        self.splits = np.array([np.nan, np.nan, np.nan])
        # splits: random/noisy, conventional, NN

        #######################
        # here we add the simulation agents
        self.addSimAgents()


    ##################################################################################
    def memory_sizes_updater(self):
        self.memory_turnover_size = round(self.replay_memory_size*self.memory_turnover_ratio)
        self.validation_set_size = self.memory_turnover_size
        

    ##################################################################################
    def generate_model(self, model_type = None):
        
        n_actions = self.n_actions_discr
            
        if model_type in ['pg','v']:
            lr= self.lr[1]
        else:
            lr= self.lr[0]
        
        if self.net_type == 'ConvModel':
                model = ConvModel(model_version = 0, net_type = self.net_type+str(self.net_version), lr= lr, \
                                  n_actions = n_actions if not (model_type == 'v') else 1, n_frames = self.n_frames, N_in = self.N_in_model, \
                                  fc_layers = self.layers_width, softmax = (model_type == 'pg') ) 

        elif self.net_type == 'ConvFrxModel':
                model = NN_frx(model_version = 0, net_type = self.net_type+str(self.net_version), lr= lr, \
                                  n_actions = n_actions if not (model_type == 'v') else 1, channels_in = 4, N_in = [self.n_frames, 2], \
                                   softmax = (model_type == 'pg') ) 
                    # conv_no_grad = ((model_type == 'pg') and self.share_conv_layers),
                # only for ConvModel in the PG case, it is allowed to "partially" update the model (convolutional layers are evaluated with "no_grad()")
        elif self.net_type == 'LinearModel': 
                #n_actions
                model = LinearModel(0, self.net_type+str(self.net_version), lr, n_actions if not (model_type == 'v') else 1 , \
                                    self.N_in_model, *(self.layers_width) , softmax = (model_type == 'pg')) 
                            #if not (model_type == 'v') else *tuple([round(l/2) for l in self.layers_width])
        return model
    
    
    ##################################################################################
    def generateDiscrActSpace_GymStyleEnv(self):
        if self.env_type  == 'RobotEnv':
            return DiscrGymStyleRobot( n_frames = self.n_frames, n_bins_act= self.discr_env_bins , sim_length = self.sim_length_max , difficulty=self.difficulty)
        elif self.env_type  == 'CartPole':
            return DiscrGymStyleCartPole(n_bins_act= self.discr_env_bins, sim_length_max = self.sim_length_max, difficulty=self.difficulty)
        elif self.env_type  == 'Platoon':
            return DiscrGymStylePlatoon(n_bins_act= self.discr_env_bins, sim_length_max = self.sim_length_max, \
                                    difficulty=self.difficulty, rewards = self.rewards, options = self.env_options)
        elif self.env_type == 'Chess':
            self.one_vs_one = True
            return GymstyleChess(use_NN = True,  max_n_moves = self.sim_length_max  , rewards = self.rewards, print_out = self.show_rendering)
        elif self.env_type == 'Connect4':
            self.one_vs_one = True
            return GymstyleConnect4(use_NN = True, rewards = self.rewards, print_out = self.show_rendering)
        elif self.env_type == 'DicePoker':
            return GymstyleDicePoker()
        elif self.env_type == 'Frx':
            env = GymstyleFrx(n_frames = self.n_frames, max_n_moves = self.sim_length_max ,initial_account = 1000)
            self.discr_env_bins = env.n_bins_act
            return env


        

    """
    ##################################################################################
    def generateContnsHybActSpace_GymStyleEnv(self):
        if self.env_type  == 'Platoon':
            return contn_hyb_GymStyle_Platoon(action_structure = [0,0,1], sim_length_max = self.sim_length_max, \
                                    difficulty=self.difficulty, rewards = self.rewards, options = self.env_options)
    """

    ##################################################################################
    #agents attributes are updated when they have the same name as the rl environment
    # except for attributes explicited in arguments
    def updateRLUpdaterAttributesExcept(self, *args):
  
        for rl_env_attr, value_out in self.__dict__.items():
            if self.ray_parallelize :
                for updater_attr in ray.get(self.nn_updater.getAttributes.remote() ):
                    if rl_env_attr == updater_attr and rl_env_attr not in args:
                            self.nn_updater.setAttribute.remote(updater_attr,value_out)
            else:
                for updater_attr in self.nn_updater.getAttributes() :
                    if rl_env_attr == updater_attr and rl_env_attr not in args:
                            self.nn_updater.setAttribute(updater_attr,value_out)

    ##################################################################################
    #agents attributes are updated when they have the same name as the rl environment
    # except for attributes explicited in arguments
    def updateAgentsAttributesExcept(self, *args, print_attribute = None):

        if self.ray_parallelize and print_attribute is not None and self.one_vs_one:
            print('update STARTED of agents with NN env')        

        # self.model_pg and self.model_qv are updated here as well        
        for rl_env_attr, value_out in self.__dict__.items():
            if self.ray_parallelize:
                for sim_ag_attr in ray.get(self.sim_agents_discr[0].getAttributes.remote() ):
                    if rl_env_attr == sim_ag_attr and rl_env_attr not in args:
                        if rl_env_attr == 'model_qv' or rl_env_attr == 'model_pg':
                            stop_here = 1
                        for agent in self.sim_agents_discr:
                            agent.setAttribute.remote(sim_ag_attr,value_out)
            else:
                for sim_ag_attr in self.sim_agents_discr[0].getAttributes():
                    if rl_env_attr == sim_ag_attr and rl_env_attr not in args:
                        for agent in self.sim_agents_discr:
                            agent.setAttribute(sim_ag_attr,value_out)


        if self.ray_parallelize and print_attribute is not None and self.one_vs_one:
            attr = ray.get(self.sim_agents_discr[0].getAttributeValue.remote(print_attribute[0]))
            if len(print_attribute)>1:
                for i in range(1,len(print_attribute)):
                    attr = attr.__dict__[print_attribute[i]]
            print('Agents (with envs) attributes update COMPLETED!')
            print(f'SIM opponent model version: {attr}')
        
        
    ##################################################################################
    def updateEpsilon(self, eps = -1):
        """ epsilon is updated at each iteration (effective for DQL only)"""
        if eps > 1 or eps <= 0:
            new_eps = np.round(self.epsilon*self.epsilon_annealing_factor, 3)
        else:
            new_eps = eps
        self.epsilon = np.maximum(new_eps, self.epsilon_min)


    ##################################################################################
    def updateCtrlr_probability(self, ctrl_prob = -1):
        if ctrl_prob > 1 or ctrl_prob <= 0:
            new_ctrl_prob = np.round(self.ctrlr_probability*self.ctrlr_prob_annealing_factor, 3)
        else:
            new_ctrl_prob = ctrl_prob
        self.ctrlr_probability = new_ctrl_prob*(new_ctrl_prob > 0.05)


    ##################################################################################
    # sim agent is created with current "model" as NN
    def addSimAgents(self):
        
        if self.ray_parallelize:
            for i in range(self.n_agents_discr):
                self.sim_agents_discr.append(Ray_SimulationAgent.remote(i, self.generateDiscrActSpace_GymStyleEnv(), rl_mode = self.rl_mode,\
                                                                    flip_gradient_sign = self.flip_grad_sign, use_reinforce = self.use_reinforce ) )  
            #for i in range(self.n_agents_contn):
            #    self.sim_agents_contn.append(Ray_SimulationAgent.remote(i, self.generateContnsHybActSpace_GymStyleEnv(), rl_mode = self.rl_mode ) )                 
        else:
            self.sim_agents_discr.append(SimulationAgent(0, self.generateDiscrActSpace_GymStyleEnv(), rl_mode = self.rl_mode, flip_gradient_sign = self.flip_grad_sign ) )  #, ,self.model
            #self.sim_agents_contn.append(SimulationAgent(0, self.generateContnsHybActSpace_GymStyleEnv(), rl_mode = self.rl_mode) )  #, ,self.model
                #, self.model, self.n_frames, self.move_to_cuda, self.net_name, self.epsilon))            
        self.updateAgentsAttributesExcept() #,'model')

    
    ##################################################################################            
    def update_net_name(self, printout = False):  
        self.net_name = self.net_type + str(self.net_version) + '_' + str(self.training_session_number)
        if printout:
            print(f'NetName : {self.net_name}')

            
    ##################################################################################
    def load(self, training_session_number = -1):
        """ load data to resume training """
 
        # first check if the folder/training log exists (otherwise start from scratch)
        # (otherwise none of the rest occurs)
 
        filename_log = os.path.join(self.storage_path, 'TrainingLog'+ '.pkl')
        if os.path.isfile(filename_log):
        
            self.log_df = pd.read_pickle(filename_log)
            if training_session_number in self.log_df['training session'].values :
                if self.log_df.shape[0] >=1:
                    # create back_up
                    filename_log = os.path.join(self.storage_path, 'TrainingLog_tsn_'+str(int(training_session_number)) + '.pkl')
                    self.log_df.to_pickle(filename_log)
                    self.log_df = self.log_df[:int(training_session_number)]
                self.training_session_number = training_session_number        
            elif training_session_number != 0 and self.log_df.shape[0] >=1:             
                # if no training session number is given, last iteration is taken
                self.training_session_number = int(self.log_df.iloc[-1]['training session'])
            else:
                self.training_session_number = training_session_number
    
            # load epsilon (never load it manually, same thing for controller percentage)
            if self.resume_epsilon and self.log_df.shape[0] >=1:
                self.epsilon = self.log_df.iloc[-1]['epsilon']
            try:
                self.ctrlr_probability = self.log_df.iloc[self.training_session_number-1]['ctrl prob']
            except Exception:
                pass
        else:
            self.training_session_number = training_session_number
                
        self.update_net_name(printout = True)
        
        # this loads models' params (both qv and pg)
        self.update_model_cpu(reload = True)
        
        # loss history file
        loss_history_file = os.path.join(self.storage_path, 'loss_history_'+ str(self.training_session_number) + '.npy')
        if os.path.isfile(loss_history_file):
            self.qval_loss_history = np.load(loss_history_file, allow_pickle=True)
            self.qval_loss_history = self.qval_loss_history[:self.training_session_number]
           
        val_history_file = os.path.join(self.storage_path, 'val_history.npy')
        if os.path.isfile(val_history_file):
            self.val_history = np.load(val_history_file)
            self.val_history = self.val_history[self.val_history[:,0]<=self.training_session_number]
        
        """
        if  self.model_qv.n_actions != self.env_discr.get_actions_structure():
            raise Exception ("Loaded model has different number of actions from current one")
        """

        # this loads the memory
        self.memory_stored = ReplayMemory(size = self.replay_memory_size, minibatch_size= self.mini_batch_size) 
        if self.rl_mode == 'DQL' and self.memory_save_load:
            self.memory_stored.load(self.storage_path, self.net_type + str(self.net_version) )
        
                                    
        if 'AC' in self.rl_mode:
            filename_pg_train = os.path.join(self.storage_path, 'PG_training'+ '.npy')
            if os.path.isfile(filename_pg_train):
                np_pg_hist = np.load(filename_pg_train)
                
                self.global_tr_sess_num= np_pg_hist[:,0].tolist()
                self.global_pg_loss    = np_pg_hist[:,1].tolist()
                self.global_pg_entropy = np_pg_hist[:,2].tolist()
                self.global_cum_reward = np_pg_hist[:,3].tolist()
                self.global_duration =   np_pg_hist[:,4].tolist()
                self.global_advantage =  np_pg_hist[:,5].tolist()


    ##################################################################################
    def check_model_version(self, print_out = False, mode = 'sim', save_v0 = False):
        """Check which model used for the simulation/update is running"""
        
        if mode == 'sim':        
            if self.rl_mode == 'DQL':
                """
                if self.ray_parallelize:
                    model_qv = ray.get(self.sim_agents_discr[0].getAttributeValue.remote('model_qv'))
                else:
                    model_qv = self.sim_agents_discr[0].getAttributeValue('model_qv')
                """    
                model_version= self.model_qv.model_version
            elif 'AC' in self.rl_mode:
                """
                if self.ray_parallelize:
                    model_pg = ray.get(self.sim_agents_discr[0].getAttributeValue.remote('model_pg'))
                    model_v = ray.get(self.sim_agents_discr[0].getAttributeValue.remote('model_v'))
                else:
                    model_pg = self.sim_agents_discr[0].getAttributeValue('model_pg')
                    model_v = self.sim_agents_discr[0].getAttributeValue('model_v')
                """
                model_version= self.model_pg.model_version
                    
            if print_out:
                print(f'SYSTEM is SIMULATING with model version: {model_version}')  
                    
            if save_v0:
                if self.rl_mode != 'AC':
                    if not os.path.isfile(os.path.join(self.storage_path,self.net_name+ '.pt')):
                        self.model_qv.save_net_params(self.storage_path,self.net_name)                
                if 'AC' in self.rl_mode:
                    if not os.path.isfile(os.path.join(self.storage_path,self.net_name+ '_policy.pt')):
                        self.model_pg.save_net_params(self.storage_path,self.net_name+'_policy')
                    if not os.path.isfile(os.path.join(self.storage_path,self.net_name+ '_state_value.pt')):
                        self.model_v.save_net_params(self.storage_path,self.net_name+'_state_value')
                        
                with open(os.path.join(self.storage_path,'train_log.txt'), 'a+') as f:
                    if self.rl_mode != 'AC':
                        f.writelines(self.net_name + "\n")
                    if 'AC' in self.rl_mode:
                        f.writelines(self.net_name+'_policy'+ "\n")
                        f.writelines(self.net_name+'_state_value'+ "\n")

        elif mode == 'update':
            if 'AC' in self.rl_mode:
                model = self.model_pg
                """
                    if self.ray_parallelize:
                        model = ray.get(self.nn_updater.getAttributeValue.remote('model_pg'))
                    else:
                        model = self.nn_updater.getAttributeValue('model_pg')
                """
            elif self.rl_mode == 'DQL':
                if self.ray_parallelize :
                    model = ray.get(self.nn_updater.getAttributeValue.remote('model_qv'))
                else:
                    model = self.nn_updater.getAttributeValue('model_qv')                
            model_version= model.model_version
            if print_out:
                print(f'MODEL UPDATED version: {model_version}')  
        return model_version


    ##################################################################################
    def update_model_cpu(self, reload = False):
        """Upload the correct model on CPU (used to select the action when a simulation is run) """
        device_cpu = torch.device('cpu')
 
        if self.rl_mode != 'AC':
            self.model_qv.load_net_params(self.storage_path,self.net_name, device_cpu)
        
        if self.one_vs_one:
            self.env_discr.update_model(self.model_qv)
       
        if 'AC' in self.rl_mode and reload: 
            # "reload" flag handles the situation where initial training has only been performed with QV network
            if not os.path.isfile(os.path.join(self.storage_path,self.net_name + '_policy.pt')):
                #self.model_pg.init_weights()
                self.model_pg.save_net_params(self.storage_path,self.net_name+ '_policy')
                #self.model_v.init_weights()
                self.model_v.save_net_params(self.storage_path,self.net_name+ '_state_value')
            else:
                self.model_pg.load_net_params(self.storage_path,self.net_name + '_policy', device_cpu)
                self.model_v.load_net_params(self.storage_path,self.net_name + '_state_value', device_cpu)
            """
                if self.share_conv_layers:
                    self.model_pg.load_conv_params(self.storage_path,self.net_name, device_cpu)
            """
        return True


    ##################################################################################
    def saveTrainIteration(self, last_iter = False):
        # we save the model at every iterations, memory only for debugging of DQL (if exlicitly requested)
        if self.rl_mode == 'DQL' and self.memory_save_load:
            if self.ray_parallelize:
                if ray.get(self.nn_updater.hasMemoryPool.remote()):
                    self.nn_updater.save_updater_memory.remote(self.storage_path, self.net_type + str(self.net_version) )
                else:
                    self.memory_stored.save(self.storage_path, self.net_type + str(self.net_version) )
            else:
                if self.nn_updater.hasMemoryPool():
                    self.nn_updater.save_updater_memory(self.storage_path, self.net_type + str(self.net_version) )
                else:
                    self.memory_stored.save(self.storage_path, self.net_type + str(self.net_version) )
        
            
        # in this case PG net is not saved by the updater, but is locally in the main thread
        if 'AC' in self.rl_mode:
            self.model_pg.save_net_params(self.storage_path,self.net_name+ '_policy')
            self.model_v.save_net_params(self.storage_path,self.net_name+ '_state_value')
            
            filename_pg_train = os.path.join(self.storage_path, 'PG_training'+ '.npy')
            #print(f'SAVING global_pg_loss. length: {len(self.global_pg_loss)}')
            
            tup = (np.array(self.global_tr_sess_num)[:,np.newaxis], np.array(self.global_pg_loss)[:,np.newaxis], \
                   np.array(self.global_pg_entropy)[:,np.newaxis], np.array(self.global_cum_reward)[:,np.newaxis], \
                   np.array(self.global_duration)[:,np.newaxis] , np.array(self.global_advantage)[:,np.newaxis]  )
            
            np_pg_hist = np.concatenate(tup,axis = 1)
            np.save(filename_pg_train, np_pg_hist)

            #self.model_qv.save_net_params(self.storage_path,self.net_name)
            
        if self.rl_mode != 'AC':
            if self.ray_parallelize :
                self.nn_updater.save_model.remote(self.storage_path,self.net_name)
            else:
                self.nn_updater.save_model(self.storage_path,self.net_name)

        filename_log = os.path.join(self.storage_path, 'TrainingLog'+ '.pkl')
        self.log_df.to_pickle(filename_log)
        
        if last_iter:
            filename_log_fin = os.path.join(self.storage_path, 'TrainingLog_tsn_'+str(self.training_session_number) + '.pkl')
            self.log_df.to_pickle(filename_log_fin)
        
        if self.qval_loss_history is not None:
            loss_history_file = os.path.join(self.storage_path, 'loss_history_'+ str(self.training_session_number) + '.npy')
            np.save(loss_history_file, self.qval_loss_history)
          
            
    ##################################################################################
    def saveValIteration(self):
          
        if self.val_history is not None:
            val_history_file = os.path.join(self.storage_path, 'val_history.npy')
            np.save(val_history_file, self.val_history )
          

    ##################################################################################
    def update_training_variables(self, partial_log, policy_loss_i, pg_entropy_i, advantage_i, temp_pg_loss, temp_entropy, temp_advantage):
        """ update lists used to display tarining progress"""
        
        self.global_tr_sess_num.append(self.training_session_number)
        self.global_pg_loss.append(policy_loss_i)
        self.global_advantage.append(advantage_i)
        self.global_pg_entropy.append(pg_entropy_i)
        self.global_cum_reward.append(np.average(partial_log[:,1]))
        self.global_duration.append(np.average(partial_log[:,0]))
        temp_pg_loss.append(policy_loss_i)
        temp_entropy.append(pg_entropy_i)
        temp_advantage.append(advantage_i)
        
        return temp_pg_loss, temp_entropy, temp_advantage


    ##################################################################################
    def runSerialized(self, memory_fill_only = False):
        """ serialized implementation of single iteration, gradient sharing"""
        
        # initial weights used for checks
        initial_pg_weights = initial_v_weights = initial_qv_weights = None
        if 'AC' in self.rl_mode:
            initial_pg_weights = deepcopy(self.model_pg.cpu().state_dict())     
            initial_v_weights = deepcopy(self.model_v.cpu().state_dict())   

        invalid_occurred = False
        total_log = None
        total_runs = 0
        qv_update_launched = False        
        
        # QV update launch on GPU
        if self.memory_stored is not None and not memory_fill_only and self.rl_mode != 'AC':
            if self.nn_updater.hasMemoryPool():

                initial_qv_weights = deepcopy(self.nn_updater.getAttributeValue('model_qv')).cpu().state_dict()     
                self.qval_loss_hist_iteration = self.nn_updater.update_DeepRL()
                qv_update_launched = False        
        
        temp_pg_loss = []
        temp_entropy = []
        temp_advantage = []
        
        while True: 
            self.display_progress(total_log)
            
            partial_log, single_runs , successful_runs, internal_memory_fill_ratio , pg_info = self.sim_agents_discr[0].run_synch()
            
            # implementation for A3C
            if 'AC' in self.rl_mode :
                grad_dict_pg, grad_dict_v, policy_loss_i, pg_entropy_i, advantage_i, valid_model = pg_info
                if valid_model:
                    invalid_occurred = False
                    temp_pg_loss, temp_entropy, temp_advantage = self.update_training_variables(partial_log, \
                            policy_loss_i, pg_entropy_i, advantage_i, temp_pg_loss, temp_entropy, temp_advantage)
                    
                    # update common model gradient
                    for net1,net2 in zip( grad_dict_pg.items() , self.model_pg.named_parameters() ):
                        net2[1].grad += net1[1].clone()
                    for net1,net2 in zip( grad_dict_v.items() , self.model_v.named_parameters() ):
                        net2[1].grad += net1[1].clone()
                        
                    # update common model (in caso of common layers gradients are transferred)
                    if self.share_conv_layers:
                        for net1,net2 in zip( self.model_pg.named_parameters() , self.model_v.named_parameters() ):
                            if net1[0] not in self.model_pg.independent_weights:
                                net1[1].grad /= 2
                                net1[1].grad += net2[1].grad.clone()/ 2
                                net2[1].grad *= 0
                    
                    self.model_pg.optimizer.step()
                    self.model_v.optimizer.step()
                    self.model_pg.optimizer.zero_grad()
                    self.model_v.optimizer.zero_grad()

                    if self.share_conv_layers:
                        for k,v in self.model_pg.state_dict().items():
                            if k not in self.model_pg.independent_weights:
                                self.model_v.state_dict()[k] *= 0
                                self.model_v.state_dict()[k] += deepcopy(self.model_pg.state_dict()[k])
                        
                    self.sim_agents_discr[0].setAttribute('model_pg',self.model_pg)
                    self.sim_agents_discr[0].setAttribute('model_v',self.model_v)
                    
                else: # invalid model
                    if invalid_occurred:
                        raise('consecutive invalid models')
                    else:
                        invalid_occurred = True

            if valid_model:
                if total_log is None:
                    total_log = partial_log
                else:
                    total_log = np.append(total_log, partial_log, axis = 0)
                total_runs += single_runs
            
            if self.rl_mode=='AC' and not self.shared_memory.isFull() and internal_memory_fill_ratio > 0.25 :
                self.shared_memory.addMemoryBatch(self.sim_agents_discr[0].emptyLocalMemory() )
                
            if (self.shared_memory.isFull() and self.rl_mode!='AC') or (self.rl_mode=='AC' and np.sum(total_log[:,0]) >= self.memory_turnover_size):
                break
        
        self.display_progress(total_log)
        print('')
        # end of iteration

        if not memory_fill_only:        
            self.post_iteration_routine(initial_qv_weights = initial_qv_weights, \
                        initial_pg_weights = initial_pg_weights, initial_v_weights= initial_v_weights, \
                        total_runs= total_runs, total_log= total_log, temp_pg_loss= temp_pg_loss, \
                        temp_entropy= temp_entropy, temp_advantage = temp_advantage)


    ##################################################################################
    def display_progress(self, total_log = None):
        """ display progress bar on terminal"""
        if self.rl_mode == 'AC':
            if total_log is None:
                progress(0, 1000)
            else:
                progress(round(1000*np.sum(total_log[:,0])/self.memory_turnover_size), 1000)
        else:
            progress(round(1000*self.shared_memory.fill_ratio), 1000 , f'Fill memory - {self.shared_memory.size}')



    ##################################################################################
    def runAC_Parallelized(self):
        """ parallelized implementation of single iteration"""
        
        # initial weights used for checks
        initial_pg_weights = initial_v_weights = initial_qv_weights = None
        if 'AC' in self.rl_mode:
            initial_pg_weights = deepcopy(self.model_pg.cpu().state_dict())  
            if self.rl_mode == 'AC':
                initial_v_weights = deepcopy(self.model_v.cpu().state_dict())     
        
        qv_update_launched = False        
        total_log = None
        total_runs = 0        
        
        temp_pg_loss = []
        temp_entropy = []
        temp_advantage = []

        task_lst = [None for _ in self.sim_agents_discr]
        task_replacement = [None for _ in self.sim_agents_discr]
        tot_epochs = 0
                
        # start of iteration
        while True:
            
            # 1) start qv update based on existing memory stored
            if self.rl_mode == 'parallelAC' and not qv_update_launched:
                if ray.get(self.nn_updater.hasMemoryPool.remote()) :
                    # this check is required to ensure nn_udater has finished loading the memory 
                    #initial_qv_weights = deepcopy(ray.get(self.nn_updater.getAttributeValue.remote('model_qv')).cpu().state_dict())     
                    initial_qv_weights = deepcopy(ray.get(self.nn_updater.getAttributeValue.remote('model_qv'))).cpu().state_dict()
                    updating_process = self.nn_updater.update_DQN_asynch.remote()
                    qv_update_launched = True

            if self.continuous_qv_update and qv_update_launched:
                model_qv_weights = ray.get(self.nn_updater.get_current_QV_model.remote()).cpu().state_dict()

            # 2) start parallel simulations
            for i, agent in enumerate(self.sim_agents_discr):
                agent.setAttribute.remote('model_pg',self.model_pg)
                if self.rl_mode == 'AC':
                    agent.setAttribute.remote('model_v',self.model_v)
                if self.continuous_qv_update and qv_update_launched:
                    agent.update_model_weights_only.remote(model_qv_weights, 'model_qv')
                task_lst[i] = agent.run.remote()
                
            n_trajectories = 0
    
            while any(tsk is not None for tsk in task_lst):
                
                self.display_progress(total_log)

                for i, agent in enumerate(self.sim_agents_discr):                
                    if not ray.get(agent.isRunning.remote()) and (task_lst[i] is not None or task_replacement[i] is not None):
                        task = task_lst[i] if task_lst[i] else task_replacement[i]

                        try:
                            partial_log, single_runs, successful_runs, internal_memory_fill_ratio , pg_info = ray.get(task)
                            grad_dict_pg, grad_dict_v, policy_loss_i, pg_entropy_i, advantage_i, valid_model = pg_info
                        except Exception:
                            valid_model = False
                            print('failed remote simulation')
                        
                        if valid_model:
                        # partial_log contains 1) duration and 2) cumulative reward of every single-run
                            if total_log is None:
                                total_log = partial_log
                            else:
                                total_log = np.append(total_log, partial_log, axis = 0)
                            total_runs += single_runs

                            temp_pg_loss, temp_entropy, temp_advantage = self.update_training_variables(partial_log, \
                                    policy_loss_i, pg_entropy_i, advantage_i, temp_pg_loss, temp_entropy, temp_advantage)
                            # update common model gradient
                            for net1,net2 in zip( grad_dict_pg.items() , self.model_pg.named_parameters() ):
                                net2[1].grad += net1[1].clone()
                            if self.rl_mode == 'AC':
                                for net1,net2 in zip( grad_dict_v.items() , self.model_v.named_parameters() ):
                                    net2[1].grad += net1[1].clone()
                        
                            if self.rl_mode != 'AC' and internal_memory_fill_ratio > 0.2:
                                self.shared_memory.addMemoryBatch( ray.get(agent.emptyLocalMemory.remote()) )
                                
                            n_trajectories += 1

                        task_replacement[i] = agent.run.remote()
                        task_lst[i] = None
                            
            for i, agent in enumerate(self.sim_agents_discr):
                if task_replacement[i] is not None:
                    agent.force_stop.remote()
                    ray.get(task_replacement[i])
                    task_replacement[i] = None
                    
            print(f'simulated trajectories: {n_trajectories}')
                            
            for net1,net2 in zip( self.model_pg.named_parameters() , self.model_v.named_parameters() ):    
                net1[1].grad/= n_trajectories
                net2[1].grad/= n_trajectories
            
            # update common model (in caso of common layers gradients are transferred)
            if self.share_conv_layers and self.rl_mode == 'AC':
                for net1,net2 in zip( self.model_pg.named_parameters() , self.model_v.named_parameters() ):
                    if net1[0] not in self.model_pg.independent_weights:
                        net1[1].grad /= 2
                        net1[1].grad += net2[1].grad.clone()/ 2
                        net2[1].grad *= 0
            
            self.model_pg.optimizer.step()
            self.model_pg.optimizer.zero_grad()

            for k,v in self.model_pg.state_dict().items():
                if torch.isnan(v).any():
                    self.unstable_model[0] = True

            if self.rl_mode == 'AC':
                self.model_v.optimizer.step()
                self.model_v.optimizer.zero_grad()

                if self.share_conv_layers and self.rl_mode == 'AC':
                    for k,v in self.model_pg.state_dict().items():
                        if k not in self.model_pg.independent_weights:
                            self.model_v.state_dict()[k] *= 0
                            self.model_v.state_dict()[k] += deepcopy(self.model_pg.state_dict()[k])
            
                for k,v in self.model_v.state_dict().items():
                    if torch.isnan(v).any():
                        self.unstable_model[0] = True
        
            tot_epochs +=1
            
            # if memory is full inform qv update it can stop
            if (self.rl_mode != 'AC' and self.shared_memory.isPartiallyFull(0.9)  ) \
                    or (self.rl_mode == 'AC' and np.sum(total_log[:,0]) >= self.memory_turnover_size  ) \
                        or self.unstable_model[0]:
                    
                if qv_update_launched:
                    self.nn_updater.sentinel.remote()
                    # after iteration extract data from remote QV update process
                    self.qval_loss_hist_iteration = np.array(ray.get(updating_process))
                break
            
        self.display_progress(total_log)
        print('')

        if self.unstable_model[0]:
            if not all(self.unstable_model):
                print('@@@@@@@@@@@@@@@@@ WARNING @@@@@@@@@@@@@@@@@@@')
                print('Issues during PG model update. Previous version is re-loaded. Simulation is aborted if problem is found in consecutive iterations.')
                print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                self.update_model_cpu(reload = True)
                self.unstable_model = [False, True]
            else:
                raise('Failed. Weights update leads to NaN model')
        else:
            self.unstable_model = [False, False]
   
        self.post_iteration_routine(initial_qv_weights = initial_qv_weights, \
                    initial_pg_weights = initial_pg_weights, initial_v_weights= initial_v_weights, \
                    total_runs= total_runs, total_log= total_log, temp_pg_loss= temp_pg_loss, \
                    temp_entropy= temp_entropy, temp_advantage = temp_advantage)


    ##################################################################################
    def runQV_Parallelized(self, memory_fill_only = False):
        """ parallelized implementation of single iteration"""
        
        initial_qv_weights = None
        qv_update_launched = False        

        total_log = None
        total_runs = 0        

        task_lst = [None for _ in self.sim_agents_discr]
        
        while True:
            # 1) start qv update based on existing memory stored
            if not memory_fill_only:
                if ray.get(self.nn_updater.hasMemoryPool.remote()) and not qv_update_launched:
                    # this check is required to ensure nn_udater has finished loading the memory 
                    initial_qv_weights = deepcopy(ray.get(self.nn_updater.getAttributeValue.remote('model_qv')).cpu().state_dict())     
                    updating_process = self.nn_updater.update_DQN_asynch.remote()
                    qv_update_launched = True
                
            # 2) start parallel simulations
            for i, agent in enumerate(self.sim_agents_discr):
                # 2.1) get info from complete processes
                if not ray.get(agent.isRunning.remote()): 
                    if task_lst[i] is not None:
                        partial_log, single_runs, successful_runs,internal_memory_fill_ratio , pg_info = ray.get(task_lst[i])
                        # partial_log contains 1) duration and 2) cumulative reward of every single-run
                        task_lst[i] = None
                        if total_log is None:
                            total_log = partial_log
                        else:
                            total_log = np.append(total_log, partial_log, axis = 0)
                        total_runs += single_runs
                        
                        self.shared_memory.addMemoryBatch(ray.get(agent.emptyLocalMemory.remote()) )
                        self.display_progress()
                    
                    # 2.1.1) run another task if required
                    expected_upcoming_memory_ratio = sum(not tsk is None for tsk in task_lst)*self.tot_iterations/self.memory_turnover_size
                    min_fill = 0.95-np.clip(expected_upcoming_memory_ratio,0,1)
                    
                    if not self.shared_memory.isPartiallyFull(min_fill):
                        if self.continuous_qv_update and qv_update_launched:
                            model_qv_weights = ray.get(self.nn_updater.get_current_QV_model.remote()).cpu().state_dict()
                            agent.update_model_weights_only.remote(model_qv_weights, 'model_qv')
                        task_lst[i] = agent.run.remote()
                    
            # if memory is (almost) full and task list empty, inform qv update it can stop
            if self.shared_memory.isPartiallyFull(0.9) and all(tsk is None for tsk in task_lst):
                if qv_update_launched:
                    self.nn_updater.sentinel.remote()
                    # after iteration extract data from remote QV update process
                    self.qval_loss_hist_iteration = np.array(ray.get(updating_process))
                break

        if not memory_fill_only:        
            self.post_iteration_routine(initial_qv_weights = initial_qv_weights, total_runs= total_runs, total_log= total_log)


    ##################################################################################
    def post_iteration_routine(self, initial_qv_weights = None, initial_pg_weights = None, initial_v_weights= None, \
                               total_runs= None, total_log= None, temp_pg_loss= None, temp_entropy= None, temp_advantage = None):
        """ store data, display intermediate results """

        average_qval_loss = policy_loss = pg_entropy = np.nan

        if initial_qv_weights is not None:
            # 1) compute qv average loss
            average_qval_loss = np.round(np.average(self.qval_loss_hist_iteration), 3)
            print(f'QV. loss = {average_qval_loss}')
            # 2) show model differences
            if self.ray_parallelize:
                current_qv_weights = deepcopy(ray.get(self.nn_updater.get_current_QV_model.remote())).cpu().state_dict()
            else:
                current_qv_weights = deepcopy(self.nn_updater.model_qv).cpu().state_dict()
            model_qv_diff = self.model_qv.compare_weights(initial_qv_weights , current_qv_weights )
            print(f'model QV update : {np.round(model_qv_diff,5)}')
            
        if initial_pg_weights is not None:
            model_diff_pg = np.round(self.model_pg.compare_weights(initial_pg_weights), 5)
            print(f'PG model change: {model_diff_pg}')
            
            if self.rl_mode == 'AC':
                model_diff_v = np.round(self.model_v.compare_weights(initial_v_weights), 5)
                print(f'V model change: {model_diff_v}')
                adv_loss = np.round(sum(temp_advantage)/(1e-5+len(temp_advantage)), 3)
                average_qval_loss = adv_loss
                
            policy_loss = np.round(sum(temp_pg_loss)/(1e-5+len(temp_pg_loss)), 3)
            pg_entropy =  np.round(sum(temp_entropy)/(1e-5+len(temp_entropy)), 3)
            
            print('')
            print(f'average policy loss : {policy_loss}')
            print(f'average pg entropy  : {np.round(100*pg_entropy/self.max_entropy, 2)}%')
            if self.rl_mode == 'AC':
                print(f'average adv loss : {adv_loss}')
            
        print('')
        print(f'average cum-reward  : {np.round(np.average(total_log[:,1]),5)}')
        #print('total log:')
        #print(total_log[:,1])
    
        #3) store computed losses
        self.get_log(total_runs, total_log, qval_loss = average_qval_loss, policy_loss = policy_loss , pg_entropy = pg_entropy)


    ##################################################################################
    def get_log(self, total_runs, total_log, **kwargs):
        
        #total log includes (for all runs): [0] steps since start [1] cumulative reward

        for k,v in kwargs.items():
            self.session_log[k] = np.round( v , 2)
        
        # total runs, steps since start, cumulative reward, average loss / total_successful_runs, entropy (pg training only)
        self.session_log['total runs'] = int(total_runs)
        self.session_log['steps since start'] = np.round(np.average(total_log[:,0]),2)
        self.session_log['cumulative reward'] = np.round(np.average(total_log[:,1]),2)

    
    ##################################################################################
    def validation_serialized(self):
        
        total_log = np.zeros((1,2),dtype = np.float)
        total_runs = 0
        tot_successful_runs = 0
            
        while not self.shared_memory.isFull():
            self.display_progress()
            
            partial_log, single_runs, successful_runs, _ , _ = self.sim_agents_discr[0].run_synch(use_NN = True)
            # partial log: steps_since_start, cum_reward
            
            total_log = np.append(total_log, partial_log, axis = 0)
            total_runs += single_runs
            tot_successful_runs += successful_runs
            
            if not self.shared_memory.isFull():
                self.shared_memory.addMemoryBatch(self.sim_agents_discr[0].emptyLocalMemory() )
                
        print('')
        print('')           
        self.end_validation_routine(total_runs,tot_successful_runs,  total_log)        
        
    
    ##################################################################################
    def validation_parallelized(self):
        
        total_log = np.zeros((1,2),dtype = np.float)
        min_fill_ratio = 0.95
        total_runs = tot_successful_runs = 0 

        while True:

            if np.array([not ray.get(agent.isRunning.remote()) for agent in self.sim_agents_discr]).all() and not self.shared_memory.isPartiallyFull(min_fill_ratio):
                agents_lst = []
                [agents_lst.append(agent.run.remote(use_NN = True)) for i, agent in enumerate(self.sim_agents_discr)]
                for agnt in agents_lst:
                    #for each single run, partial log includes: 1) steps since start 2) cumulative reward
                    partial_log, single_runs, successful_runs, _ , _ = ray.get(agnt)
                    total_log = np.append(total_log, partial_log, axis = 0)
                    total_runs += single_runs
                    tot_successful_runs += successful_runs
            
            if not self.shared_memory.isPartiallyFull(min_fill_ratio):
                for agent in self.sim_agents_discr:
                    self.shared_memory.addMemoryBatch(ray.get(agent.emptyLocalMemory.remote()) )
                    self.display_progress()
    
            if self.shared_memory.isPartiallyFull(min_fill_ratio): # or break_cycle:
                break
            
        print('')
        self.end_validation_routine(total_runs, tot_successful_runs, total_log)


    ##################################################################################
    def end_validation_routine(self, total_runs, tot_successful_runs, total_log):
        
        new_memory = self.shared_memory.getMemoryAndEmpty()
        turnover_pctg_act = np.round(100*self.memory_stored.addMemoryBatch(new_memory),2)
        if self.ray_parallelize :
            self.nn_updater.update_memory_pool.remote(new_memory)
        else:
            self.nn_updater.update_memory_pool(new_memory)
        
        #total log includes (for all runs): 1) steps since start 2) cumulative reward
        self.get_log(total_runs, total_log)
        if total_runs > 0 :
            success_ratio = np.round(tot_successful_runs/total_runs ,2)
        else:
            success_ratio = -1

        val_history = np.array([self.training_session_number, success_ratio, self.session_log['total runs'], \
                                self.session_log['steps since start'], self.session_log['cumulative reward'] ]) 
            
        # 0) iteration ---- 1)success_ratio  2) total runs 3) average duration   4)average single run reward           
        if self.val_history is None:
            self.val_history =  val_history[np.newaxis,:].copy()
        else:
            self.val_history = np.append(self.val_history, val_history[np.newaxis,:], axis = 0)
            
        if self.bashplot and self.val_history.shape[0] > 5:
            np.savetxt("val_hist.csv", self.val_history[:,[0,3]], delimiter=",")
            plot_scatter(f = "val_hist.csv",xs = None, ys = None, size = 20, colour = 'yellow',pch = '*', title = 'validation history')
            
        self.saveValIteration()
        
        print(f'training iteration         = {val_history[0]}')
        print(f'test samples               = {round(self.shared_memory.fill_ratio*self.shared_memory.size)}')
        print(f'total single runs          = {val_history[2]}')
        print(f'success ratio              = {val_history[1]}')
        print(f'average steps since start  = {val_history[3]}')
        print(f'average cumulative reward  = {val_history[4]}')


    ##################################################################################
    def print_NN_parameters_count(self):
        
        print('')
        print('')
        
        if self.rl_mode == 'AC':
            actor_n_params  = self.model_pg.count_NN_params()
            critic_n_params = self.model_v.count_NN_params()
            print('total NN trainable parameters:')
            print(f'actor  :{actor_n_params}')
            print(f'critic :{critic_n_params}')
        elif self.rl_mode == 'parallelAC':
            actor_n_params  = self.model_pg.count_NN_params()
            QV_critic_n_params = self.model_qv.count_NN_params()
            print('total NN trainable parameters:')
            print(f'actor      :{actor_n_params}')
            print(f'critic (QV):{QV_critic_n_params}')
        elif self.rl_mode == 'DQL':
            DQL_n_params = self.model_qv.count_NN_params()
            print('total NN trainable parameters:')
            print(f'DQL :{DQL_n_params}')
        
        print('')
        print('')


    ##################################################################################
    def runSequence(self, n_runs = 5, display_graphs = False, reset_optimizer = False):
        
        self.print_NN_parameters_count()
        
        final_run = self.training_session_number+ n_runs
        
        for i in np.arange(self.training_session_number,final_run+1):
            
            self.runEnvIterationOnce(reset_optimizer)
            
            print('###################################################################')
            print('###################################################################')
            print(f'Model Saved in {self.storage_path}')
            print(f'end of iteration: {self.training_session_number -1} of {final_run}')
            print('###################################################################')
            print('###################################################################')
            
            if reset_optimizer:
                reset_optimizer = False
            if display_graphs:
                self.plot_training_log()
                
            if i> 1 and not i % self.val_frequency and self.rl_mode == 'DQL':
                print('###################################################################')
                print('validation cycle start...')
                self.shared_memory.resetMemory(self.validation_set_size)
                if self.ray_parallelize:
                    self.validation_parallelized()
                else:
                    self.validation_serialized()
                print('end of validation cycle')
                print('###################################################################')


    ##################################################################################
    def updateProbabilities(self, print_out = False):

        self.updateEpsilon()
        self.updateCtrlr_probability()
        
        split_conventional_ctrl = np.round(100*self.ctrlr_probability,2)
        split_random = np.round(100*self.epsilon*(1-self.ctrlr_probability),2)
        split_NN_ctrl = np.round(100*(1-self.epsilon)*(1-self.ctrlr_probability),2)
        
        self.splits = np.array([split_random, split_conventional_ctrl, split_NN_ctrl])
        
        if print_out:

            print(f'y = {self.epsilon}')
            print(f'c = {self.ctrlr_probability}')

            print(f'Random share: {split_random}%')
            print(f'Conventional ctl share: {split_conventional_ctrl}%')
            print(f'NN share: {split_NN_ctrl}%')
        

    ##################################################################################
    def pre_training_routine(self, reset_optimizer = False, first_pg_training = False):
        # it only depends on the memory size
        
        # shared memory is the one filled by the agents, to be poured later in "stored memory".
        #it's resetted at every new cycle. 
        #after the first fill, only the portion defined by "memory_turnover_size" is filled
        self.shared_memory.resetMemory(bool(self.memory_stored)*self.memory_turnover_size)
        self.updateProbabilities(print_out = True)
        
        # update epsilon and model of the Sim agents
        if self.training_session_number == 0 or first_pg_training:
            self.check_model_version(print_out = True, mode = 'sim', save_v0 = True)

        #self.update_model_cpu() returns True if succesful
        if self.update_model_cpu():
            self.updateAgentsAttributesExcept(print_attribute=['env','env','model','model_version'])
        else:
            raise('Loading error')
        
        if self.rl_mode != 'AC':
            if self.nn_updater is None:
                # if needed, initialize the model for the updating
                self.initialize_updater(reset_optimizer)
                if bool(self.memory_stored):
                    # if memory stored is present, it means memory has been loaded
                    self.load_memorystored_to_updater()
    
            else:
                # only update the net name (required to save/load net params)
                if self.ray_parallelize :
                    self.nn_updater.setAttribute.remote('net_name',self.net_name)
                else:                
                    self.nn_updater.setAttribute('net_name',self.net_name)
        
        
    ##################################################################################
    def initialize_updater(self, reset_optimizer = False):        
        # instanciate ModelUpdater actor -> everything here happens on GPU
        # initialize models (will be transferred to NN updater)
        model_qv = self.generate_model('qv')
            
        if self.ray_parallelize :
            self.nn_updater = Ray_RL_Updater.remote()
            self.updateRLUpdaterAttributesExcept()
            self.nn_updater.load_model.remote(model_qv,  reset_optimizer=reset_optimizer)
        else:
            self.nn_updater = RL_Updater()
            self.updateRLUpdaterAttributesExcept()
            self.nn_updater.load_model(model_qv, reset_optimizer=reset_optimizer)
        
        
    ##################################################################################
    def q_val_bashplot(self):
        if self.bashplot and self.qval_loss_history is not None:       
            #print(f'history length= {self.loss_history_full.shape[0]}')
            if self.qval_loss_history.shape[0] > 2500:
                np.savetxt("loss_hist.csv", np.concatenate((np.arange(1,self.qval_loss_history[-2000:].shape[0]+1)[:,np.newaxis],self.qval_loss_history[-2000:,np.newaxis]), axis = 1), delimiter=",")
                plot_scatter(f = "loss_hist.csv",xs = None, ys = None, size = 20, colour = 'white',pch = '*', title = 'training loss')
        
        
    ##################################################################################
    def load_memorystored_to_updater(self):            
        if self.ray_parallelize :
            self.nn_updater.setAttribute.remote('memory_pool', self.memory_stored)
        else:
            self.nn_updater.setAttribute('memory_pool', self.memory_stored)

        
    ##################################################################################
    def end_training_round_routine(self):
        
        # update memory
        if self.rl_mode != 'AC':
        
            if self.memory_stored is None:
                self.memory_stored = self.shared_memory.cloneMemory()
                self.load_memorystored_to_updater()
                turnover_pctg_act = np.round(100*self.memory_stored.fill_ratio,2)
            else:
                new_memory = self.shared_memory.getMemoryAndEmpty()
                turnover_pctg_act = np.round(100*self.memory_stored.addMemoryBatch(new_memory),2)
                if self.ray_parallelize :
                    self.nn_updater.update_memory_pool.remote(new_memory)
                else:
                    self.nn_updater.update_memory_pool(new_memory)
            
        #print(f'memory size = {self.memory_stored.size*self.memory_stored.fill_ratio}')
        #print(f'effective memory turnover = {turnover_pctg_act}')
            
        # update log and history
        # (log structure: ['training session','epsilon', 'total single-run(s)','average single-run duration',\
        #    'average single-run reward' , 'average qval loss','average policy loss', 'N epochs', 'memory size', \
        #    action gen splits (0,1,2)..., 'pg_entropy']  )
        if self.training_session_number > 1:
            new_row = [int(self.training_session_number), self.epsilon, self.session_log['total runs'] , \
                       self.session_log['steps since start'] , self.session_log['cumulative reward'], self.session_log['qval_loss'],\
                           self.session_log['policy_loss'], self.n_epochs, self.replay_memory_size, \
                               self.splits[0], self.splits[1], self.splits[2] ,  \
                                   self.session_log['pg_entropy']   ]
                
            self.log_df.loc[len(self.log_df)] = new_row
        
        if self.qval_loss_history is None:
            self.qval_loss_history = self.qval_loss_hist_iteration
        else:
            self.qval_loss_history = np.append(self.qval_loss_history, self.qval_loss_hist_iteration)

        self.q_val_bashplot()

        # models and log are saved at each iteration!
        self.saveTrainIteration()


    ##################################################################################
    def runEnvIterationOnce(self, reset_optimizer = False):
        
        # loading/saving of models, verifying correct model is used at all stages
        self.pre_training_routine(reset_optimizer)
        
        # core functionality
        if not self.ray_parallelize:
            self.runSerialized()
        else:
            # run simulations/training in parallel
            if self.rl_mode == 'DQL':
                self.runQV_Parallelized()
            else:
                self.runAC_Parallelized()
        
        # we update the name in order to save it
        self.training_session_number +=1        
        self.update_net_name()
        
        self.end_training_round_routine()
       
        
    ##################################################################################
    # plot the training history
    def plot_training_log(self, init_epoch=0, qv_loss_log = False, pg_loss_log = False):
        #log_norm_df=(self.log_df-self.log_df.min())/(self.log_df.max()-self.log_df.min())
        #fig, ax = plt.subplots()
        # (log structure: 
        # 0: 'training session', 1: 'epsilon', 2: 'total single-run(s)',
        # 3: 'average single-run duration', 4: 'average single-run reward' , 5: 'av. q-val loss', 
        # 6: 'av. policy loss', 7: 'N epochs', 8: 'memory size', 9: 'split random/noisy',
        # 10:'split conventional', 11: 'split NN', 12: 'pg_entropy'] )        

        indicators = self.log_df.columns

        ######################################
        # figsize=(12, 12)
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        
        # 'total single-run(s)'
        ax1.plot(self.log_df.iloc[init_epoch:][indicators[3]])
        ax1.legend([indicators[3]])
        
        # 'average single-run duration'
        ax2.plot(self.log_df.iloc[init_epoch:][indicators[4]])
        ax2.legend([indicators[4]])
        
        # train splits
        self.log_df.iloc[init_epoch:][['split random/noisy','split conventional','split NN']].plot.area(stacked=True, ax = ax3)
        
        ######################################
        fig_1 = plt.figure()
        
        # 'av. q-val loss'
        ax1_1 = fig_1.add_subplot(411)
        ax1_1.plot(self.log_df.iloc[init_epoch:][indicators[5]])
        if qv_loss_log:
            ax1_1.set_yscale('log')
        ax1_1.legend(['q-val (AC) / s-val (DQL) loss'])

        # 'av. policy loss'
        ax2_1 = fig_1.add_subplot(412)
        ax2_1.plot(self.log_df.iloc[init_epoch:][indicators[6]])
        if pg_loss_log:
            ax2_1.set_yscale('symlog')
        ax2_1.legend([indicators[6]])
        
        # 'pg entropy'
        ax3_1 = fig_1.add_subplot(413)
        ax3_1.plot(self.log_df.iloc[init_epoch:][indicators[12]])
        ax3_1.legend([indicators[12]])

        # 'average single-run reward'
        ax4_1 = fig_1.add_subplot(414)
        ax4_1.plot(self.log_df.iloc[init_epoch:][indicators[4]])
        ax4_1.legend([indicators[4]])
    
        if 'AC' in self.rl_mode:

            fig_2, ax2 = plt.subplots(5,1)

            ax2[0].plot(self.global_pg_loss[init_epoch:])
            ax2[0].legend(['pg loss complete'])
            
            ax2[1].plot(self.global_pg_entropy[init_epoch:])
            ax2[1].legend(['entropy complete'])

            ax2[2].plot(self.global_cum_reward[init_epoch:])
            ax2[2].legend(['av. cum reward'])
            
            ax2[3].plot(self.global_duration[init_epoch:])
            ax2[3].legend(['av. duration'])

            ax2[4].plot(self.global_advantage[init_epoch:])
            ax2[4].legend(['advantage'])
            
            return fig, fig_1 , fig_2
    
        return fig, fig_1 

#%%

if __name__ == "__main__":
    #env = SimulationAgent(0, env = disc_GymStyle_Robot(n_bins_act=4), model = ConvModel())
    #rl_env = Multiprocess_RL_Environment('RobotEnv' , 'ConvModel')
    rl_env = Multiprocess_RL_Environment('CartPole' , 'LinearModel0')
