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

csfp = os.path.abspath(os.path.dirname(__file__))
if csfp not in sys.path:
    sys.path.insert(0, csfp)
    
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
from envs.gymstyle_envs import discr_GymStyle_Robot, discr_GymStyle_CartPole, discr_GymStyle_Platoon, \
    contn_hyb_GymStyle_Platoon, Gymstyle_Chess, Gymstyle_Connect4, Gymstyle_DicePoker
from nns.custom_networks import ConvModel, LinearModel
from .resources.memory import ReplayMemory
from .resources.sim_agent import SimulationAgent
from .resources.updater import RL_Updater 
from .resources.parallelize import Ray_RL_Updater , Ray_SimulationAgent

from utilities import progress, lineno, check_WhileTrue_timeout

#%%

DEBUG_MODEL_VERSIONS = True


#%%
class Multiprocess_RL_Environment:
    def __init__(self, env_type , net_type , net_version ,rl_mode = 'DQL', n_agents = 1, ray_parallelize = False , \
                 move_to_cuda = True, show_rendering = False , n_frames=4, learning_rate = 0.001 , \
                 rl_params = None,movie_frequency = 20, max_steps_single_run = 200, \
                 training_session_number = None, save_movie = True, live_plot = False, \
                 display_status_frequency = 50, max_consecutive_env_fails = 3, tot_iterations = 400,\
                 epsilon = 1, epsilon_annealing_factor = 0.95,  \
                 epsilon_min = 0.01 , gamma = 0.99 , discr_env_bins = 10 , N_epochs = 1000, \
                 replay_memory_size = 10000, mini_batch_size = 512, sim_length_max = 100, \
                 ctrlr_prob_annealing_factor = .9,  ctrlr_probability = 0, difficulty = 0, \
                 memory_turnover_ratio = 0.2, val_frequency = 10, bashplot = False, layers_width= (5,5), \
                 rewards = np.ones(5), validation_set_ratio = 4, env_options = {}, pg_partial_update = False, 
                 beta_PG = 1, n_epochs_PG = 100, batch_size_PG = 32, noise_factor = 1):
        
        self.update_policy_online = True
        self.one_vs_one = False
        
        if self.update_policy_online:
            self.global_pg_loss = []
            self.global_pg_entropy = []

        
        ####################### net/environment initialization
        # check if env/net types make sense
        allowed_envs = ['RobotEnv', 'CartPole', 'Platoon', 'Chess', 'Connect4', 'DicePoker']
        allowed_nets = ['ConvModel', 'LinearModel'] #, 'LinearModel0', 'LinearModel1', 'LinearModel2', 'LinearModel3', 'LinearModel4']
        allowed_rl_modes = ['DQL', 'AC']
        
        self.net_version = net_version
        
        if net_type not in allowed_nets or env_type not in allowed_envs or rl_mode not in allowed_rl_modes:
            raise('Env/Net Type not known')
            
        # net type has to be declared at every instanciation. net_type + ts_number define the net_name        
        self.net_type = net_type
        self.env_type = env_type
        self.rl_mode = rl_mode
        
        self.training_session_number = 0 #by default ==0, if saved net is loaded this will be changed
        self.update_net_name()
        
        #######################
        # training options
        self.bashplot = bashplot 
        self.difficulty = difficulty
        self.sim_length_max = sim_length_max # seconds after which simulation is over (and successful)
        self.lr = learning_rate
        self.val_frequency = val_frequency
        self.ray_parallelize = ray_parallelize
        self.resume_epsilon = False
        self.pg_partial_update = pg_partial_update
        
        ####################### storage options        
        # storage path changes for every model
        self.storage_path = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) ,"Data" , self.env_type, self.net_type+str(self.net_version) )
        createPath(self.storage_path).mkdir(parents=True, exist_ok=True)

        #######################
        # training parameters
        self.gamma = gamma
        self.n_epochs = N_epochs
        self.mini_batch_size = mini_batch_size
        self.memory_turnover_ratio = memory_turnover_ratio
        two_power = [2**i for i in range(10)]
        self.mb_policy = round(self.mini_batch_size / two_power[np.abs(round(1/self.memory_turnover_ratio) - np.array(two_power)).argmin()-1])
        self.replay_memory_size = replay_memory_size
        self.validation_set_ratio = validation_set_ratio # relative dimension of validation set to test_set (n times smaller)
        self.memory_sizes_updater()
        
        self.beta_PG       = beta_PG
        self.n_epochs_PG   = n_epochs_PG
        self.batch_size_PG = batch_size_PG
        
        ####################### required to set up sim_agents agents created at the end)
        # NN options
        self.move_to_cuda = torch.cuda.is_available() and move_to_cuda
        
        if self.net_type == 'ConvModel': # if we don't use conv_net there is no reason to use multichannel structure
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
        
        # added after configuration save (these attributes can't be saved in JSON format)
        self.env_discr = self.generateDiscrActSpace_GymStyleEnv()
        # this env can be used as reference and to get topologic information, but is not used for simulations!
        #(those are inside each agent)
        self.n_actions_discr = self.env_discr.get_actions_structure()
        self.noise_sd = noise_factor/np.clip(self.n_actions_discr, 5, 20)
        
        self.N_in_model = self.env_discr.get_observations_structure()

        self.model_qv = self.generate_model()
        self.model_qv.init_weights()
        
        if self.use_continuous_act_env:
            self.env_contn = self.generateContnsHybActSpace_GymStyleEnv()
            self.n_actions_contn = self.env_contn.get_actions_structure()

        self.model_pg = self.generate_model(pg_model=True, )
        self.model_pg.init_weights()
        
        # in one vs. one the environment requires the (latest) model to work
        if self.one_vs_one:
            self.env_discr.update_model(self.model_qv)
        
        # this part is basically only needed for 1vs1 (environment has to be updated within sim-agents)
        if self.use_continuous_act_env:
            self.env = self.env_contn
        else:
            self.env = self.env_discr

        #######################
        # Replay Memory init
        self.shared_memory = ReplayMemory(size = self.replay_memory_size, minibatch_size= self.mini_batch_size)
        
        #######################
        # empty initializations
        self.qval_loss_history = self.qval_loss_hist_iteration = self.val_history = None
        self.nn_updater = None
        self.memory_stored = None
        self.policy_loaded = False

        self.log_df = pd.DataFrame(columns=['training session','epsilon', 'ctrl prob','total single-run(s)',\
                                    'average single-run duration', 'average single-run reward' , 'av. q-val loss', \
                                    'av. policy loss','N epochs', 'memory size', 'split random/noisy','split conventional',\
                                        'split NN', 'pg_entropy'])
        self.splits = np.array([np.nan, np.nan, np.nan])
        # splits: random/noisy, conventional, NN

        #######################
        # here we add the simulation agents
        self.addSimAgents()


    ##################################################################################
    def memory_sizes_updater(self):
        self.memory_turnover_size = round(self.replay_memory_size*self.memory_turnover_ratio)
        self.validation_set_size = round(self.replay_memory_size/self.validation_set_ratio)
        

    ##################################################################################
    def generate_model(self, pg_model = False):
        
        if self.use_continuous_act_env:
            n_actions = self.n_actions_contn
        else:
            n_actions = self.n_actions_discr
        
        if self.net_type == 'ConvModel':
                model = ConvModel(model_version = 0, net_type = self.net_type+str(self.net_version), lr= self.lr, \
                                  n_actions = n_actions, channels_in = self.n_frames, N_in = self.N_in_model,
                                  conv_no_grad = (pg_model and self.pg_partial_update), softmax = pg_model)
                # only for ConvModel in the PG case, it is allowed to "partially" update the model (convolutional layers are evaluated with "no_grad()")
        elif self.net_type == 'LinearModel': 
                model = LinearModel(0, self.net_type+str(self.net_version), self.lr, n_actions, \
                                    self.N_in_model, *(self.layers_width), softmax = pg_model) 
        return model
    
    
    ##################################################################################
    def generateDiscrActSpace_GymStyleEnv(self):
        if self.env_type  == 'RobotEnv':
            return discr_GymStyle_Robot(n_bins_act= self.discr_env_bins)
        elif self.env_type  == 'CartPole':
            return discr_GymStyle_CartPole(n_bins_act= self.discr_env_bins, sim_length_max = self.sim_length_max, difficulty=self.difficulty)
        elif self.env_type  == 'Platoon':
            return discr_GymStyle_Platoon(n_bins_act= self.discr_env_bins, sim_length_max = self.sim_length_max, \
                                    difficulty=self.difficulty, rewards = self.rewards, options = self.env_options)
        elif self.env_type == 'Chess':
            self.one_vs_one = True
            return Gymstyle_Chess(use_NN = True,  max_n_moves = self.sim_length_max  , rewards = self.rewards, print_out = self.show_rendering)
        elif self.env_type == 'Connect4':
            self.one_vs_one = True
            return Gymstyle_Connect4(use_NN = True, rewards = self.rewards, print_out = self.show_rendering)
        elif self.env_type == 'DicePoker':
            return Gymstyle_DicePoker()

        
    ##################################################################################
    def generateContnsHybActSpace_GymStyleEnv(self):
        if self.env_type  == 'Platoon':
            return contn_hyb_GymStyle_Platoon(action_structure = [0,0,1], sim_length_max = self.sim_length_max, \
                                    difficulty=self.difficulty, rewards = self.rewards, options = self.env_options)


    ##################################################################################
    #agents attributes are updated when they have the same name as the rl environment
    # except for attributes explicited in arguments
    def updateRLUpdaterAttributesExcept(self, *args):
  
        for rl_env_attr, value_out in self.__dict__.items():
            if self.ray_parallelize:
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
                self.sim_agents_discr.append(Ray_SimulationAgent.remote(i, self.generateDiscrActSpace_GymStyleEnv(), rl_mode = self.rl_mode ) )  
            for i in range(self.n_agents_contn):
                self.sim_agents_contn.append(Ray_SimulationAgent.remote(i, self.generateContnsHybActSpace_GymStyleEnv(), rl_mode = self.rl_mode ) )                 
        else:
            self.sim_agents_discr.append(SimulationAgent(0, self.generateDiscrActSpace_GymStyleEnv(), rl_mode = self.rl_mode) )  #, ,self.model
            #self.sim_agents_contn.append(SimulationAgent(0, self.generateContnsHybActSpace_GymStyleEnv(), rl_mode = self.rl_mode) )  #, ,self.model
                #, self.model, self.n_frames, self.move_to_cuda, self.net_name, self.epsilon))            
        self.updateAgentsAttributesExcept() #,'model')

    
    ##################################################################################            
    def update_net_name(self, printout = False):  
        self.net_name = self.net_type + str(self.net_version) + '_' + str(self.training_session_number)
        if printout:
            print(f'NetName : {self.net_name}')

            
    ##################################################################################
    def load(self, training_session_number = -1, load_memory = True):
 
        # first check if the folder/training log exists (otherwise start from scratch)
        # (otherwise none of the rest occurs)
 
        filename_log = os.path.join(self.storage_path, 'TrainingLog'+ '.pkl')
        if os.path.isfile(filename_log):
        
            self.log_df = pd.read_pickle(filename_log)
            if training_session_number in self.log_df['training session'].values:
                # create back_up
                filename_log = os.path.join(self.storage_path, 'TrainingLog_tsn_'+str(int(training_session_number)) + '.pkl')
                self.log_df.to_pickle(filename_log)
                self.log_df = self.log_df[:int(training_session_number)]
                self.training_session_number = training_session_number        
            elif training_session_number != 0:             
                # if no training session number is given, last iteration is taken
                self.training_session_number = int(self.log_df.iloc[-1]['training session'])
    
            # load epsilon (never load it manually, same thing for controller percentage)
            if self.resume_epsilon:
                self.epsilon = self.log_df.iloc[-1]['epsilon']
            try:
                self.ctrlr_probability = self.log_df.iloc[self.training_session_number-1]['ctrl prob']
            except Exception:
                pass
                
            self.update_net_name(printout = True)
            
            # this loads models' params (both qv and pg)
            #self.model_qv.cpu()
            self.update_model_cpu()
            
            # loss history file
            loss_history_file = os.path.join(self.storage_path, 'loss_history_'+ str(self.training_session_number) + '.npy')
            if os.path.isfile(loss_history_file):
                self.qval_loss_history = np.load(loss_history_file)
                self.qval_loss_history = self.qval_loss_history[:self.training_session_number]
               
            val_history_file = os.path.join(self.storage_path, 'val_history.npy')
            if os.path.isfile(val_history_file):
                self.val_history = np.load(val_history_file)
                self.val_history = self.val_history[self.val_history[:,0]<=self.training_session_number]
                
            if  self.model_qv.n_actions != self.env_discr.get_actions_structure():
                raise Exception ("Loaded model has different number of actions from current one")

            # this loads the memory
            self.memory_stored = ReplayMemory(size = self.replay_memory_size, minibatch_size= self.mini_batch_size) 
            if load_memory:
                if os.path.isfile(os.path.join(self.storage_path,self.net_name+'_memory.pt')):
                    self.memory_stored.load(self.storage_path,self.net_name)
                    self.replay_memory_size = self.memory_stored.size
                    self.memory_sizes_updater()
                else:
                    print('creating memory (and saving it to be sure)')
                    self.check_model_version(print_out = True, mode = 'sim', save_v0 = True)
                    if self.ray_parallelize:
                        self.runParallelized(memory_fill_only = True)
                    else:
                        self.runSerialized(memory_fill_only = True)
                    self.memory_stored = self.shared_memory.cloneMemory()
                    self.memory_stored.save(self.storage_path,self.net_name)

                                        
            if self.update_policy_online:
                filename_pg_train = os.path.join(self.storage_path, 'PG_training'+ '.npy')
                np_pg_hist = np.load(filename_pg_train)[:self.training_session_number]
                
                self.global_pg_loss    = np_pg_hist[:,0].tolist()
                self.global_pg_entropy = np_pg_hist[:,1].tolist()


    ##################################################################################
    def check_model_version(self, print_out = False, mode = 'sim', save_v0 = False):
        """Check which model used for the simulation/update is running"""
        
        if mode == 'sim':        
            if self.use_continuous_act_env:
                if self.rl_mode == 'DQL':
                    if self.ray_parallelize:
                        model = ray.get(self.sim_agents_contn[0].getAttributeValue.remote('model_qv'))
                    else:
                        model = self.sim_agents_contn[0].getAttributeValue('model_qv')
                elif self.rl_mode == 'AC':
                    if self.ray_parallelize:
                        model = ray.get(self.sim_agents_contn[0].getAttributeValue.remote('model_pg'))
                    else:
                        model = self.sim_agents_contn[0].getAttributeValue('model_pg')
            else:
                if self.rl_mode == 'DQL':
                    if self.ray_parallelize:
                        model = ray.get(self.sim_agents_discr[0].getAttributeValue.remote('model_qv'))
                    else:
                        model = self.sim_agents_discr[0].getAttributeValue('model_qv')
                elif self.rl_mode == 'AC':
                    if self.ray_parallelize:
                        model = ray.get(self.sim_agents_discr[0].getAttributeValue.remote('model_pg'))
                    else:
                        model = self.sim_agents_discr[0].getAttributeValue('model_pg')
            
            model_v= model.model_version
                    
            if save_v0:
                if not os.path.isfile(os.path.join(self.storage_path,self.net_name+ '.pt')):
                    model.save_net_params(self.storage_path,self.net_name)                
                if self.rl_mode == 'AC':
                    if not os.path.isfile(os.path.join(self.storage_path,self.net_name+ '_policy.pt')):
                        model.save_net_params(self.storage_path,self.net_name+'_policy')
                        
                with open(os.path.join(self.storage_path,'train_log.txt'), 'a+') as f:
                    f.writelines(self.net_name + "\n")
                    if self.rl_mode == 'AC':
                        f.writelines(self.net_name+'_policy'+ "\n")
                    
            if print_out:
                print(f'SYSTEM is SIMULATING with model version: {model_v}')  

        elif mode == 'update':
            if self.rl_mode == 'AC':
                if self.ray_parallelize:
                    model = ray.get(self.nn_updater.getAttributeValue.remote('model_pg'))
                else:
                    model = self.nn_updater.getAttributeValue('model_pg')
            elif self.rl_mode == 'DQL':
                if self.ray_parallelize:
                    model = ray.get(self.nn_updater.getAttributeValue.remote('model_qv'))
                else:
                    model = self.nn_updater.getAttributeValue('model_qv')                
                    
            model_v= model.model_version

            if print_out:
                print(f'MODEL UPDATED version: {model_v}')  
                #print(f'fc1.weight (first 5 rows)= {model.fc1.weight[:5,:]}')

        return model_v


    ##################################################################################
    def update_model_cpu(self):
        """Upload the correct model on CPU (used to select the action when a simulation is run) """
        device_cpu = torch.device('cpu')
 
        self.model_qv.load_net_params(self.storage_path,self.net_name, device_cpu)
        
        if self.one_vs_one:
            self.env_discr.update_model(self.model_qv)
       
        if self.rl_mode == 'AC': 
            #if os.path.isfile(os.path.join(self.storage_path,self.net_name + '_policy.pt')):
            self.model_pg.load_net_params(self.storage_path,self.net_name + '_policy', device_cpu)
            self.policy_loaded = True
        return True

    ##################################################################################
    def saveTrainIteration(self, last_iter = False):
        # we save the model at every iterations, memory only at the end (not inside this function)
        
        if self.ray_parallelize:
            # save generated model (parallelized version)
            self.nn_updater.save_model.remote(self.storage_path,self.net_name)
        else:
            self.nn_updater.save_model(self.storage_path,self.net_name)
            
        # in this case PG net is not saved by the updater, but is locally in the main thread
        if self.update_policy_online:
            self.model_pg.save_net_params(self.storage_path,self.net_name+ '_policy')
            filename_pg_train = os.path.join(self.storage_path, 'PG_training'+ '.npy')
            np_pg_hist = np.concatenate((np.array(self.global_pg_loss)[:,np.newaxis], np.array(self.global_pg_entropy)[:,np.newaxis]),axis = 1)
            np.save(filename_pg_train, np_pg_hist)


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
    def runSerialized(self, memory_fill_only = False):

        initial_pg_weights = deepcopy(self.model_pg.cpu().state_dict())        

        average_qval_loss = policy_loss = pg_entropy = np.nan
        total_log = np.zeros((1,2),dtype = np.float)
        total_runs = 0
        
        if DEBUG_MODEL_VERSIONS:
            model_version_sim = self.check_model_version(print_out = True, mode = 'sim', save_v0 = False)
                
        if self.memory_stored is not None and not memory_fill_only: # and not self.model_new.is_updating:
            #print('entered NN update')
            self.qval_loss_hist_iteration, _ = np.array(self.nn_updater.update_DeepRL())
            average_qval_loss = np.average(self.qval_loss_hist_iteration)
            
        while not self.shared_memory.isFull():
            progress(round(1000*self.shared_memory.fill_ratio), 1000 , 'Fill memory')
            partial_log, single_runs , successful_runs, pg_info = self.sim_agents_discr[0].run_synch()
            
            
            # implementation for A3C
            if self.rl_mode == 'AC' and self.update_policy_online:
                delta_theta_i, policy_loss, pg_entropy, valid_model = pg_info
                self.global_pg_loss.append(policy_loss)
                self.global_pg_entropy.append(pg_entropy)
            """
                if valid_model:
                    for k in self.model_pg.state_dict().keys():
                        self.model_pg.state_dict()[k] += delta_theta_i[k]
                    self.sim_agents_discr[0].setAttribute('model_pg',self.model_pg)
            """    
            
            total_log = np.append(total_log, partial_log, axis = 0)
            total_runs += single_runs
            
            if not self.shared_memory.isFull():
                self.shared_memory.addMemoryBatch(self.sim_agents_discr[0].emptyLocalMemory() )
        
        progress(round(1000*self.shared_memory.fill_ratio), 1000 , 'Fill memory')
                
        """ will be removed from here"""
        # implementation for A3C
        if self.rl_mode == 'AC' and self.update_policy_online:
            if valid_model:
                self.model_pg = self.sim_agents_discr[0].getAttributeValue('model_pg')
        """ will be removed from here"""    

                
        if not memory_fill_only:
            if self.rl_mode == 'AC' and self.policy_loaded and not self.update_policy_online:
                if DEBUG_MODEL_VERSIONS:
                    model_version_update = self.check_model_version(print_out = False, mode = 'update', save_v0 = False)
                    print(f'model_version_sim = {model_version_sim}')
                    print(f'model_version_update = {model_version_update}')
                policy_loss, pg_entropy =  self.nn_updater.update_DeepRL('policy', self.shared_memory.memory ) 
                
            elif self.rl_mode == 'AC' and self.update_policy_online:
                model_diff = np.round(self.model_pg.compare_weights(initial_pg_weights), 5)
                print(f'PG model change: {model_diff}')
                
            self.get_log(total_runs, total_log, average_qval_loss, policy_loss , pg_entropy)
    

    ##################################################################################
    def runParallelized(self, memory_fill_only = False):
        
        initial_pg_weights = deepcopy(self.model_pg.cpu().state_dict())
        
        qv_update_launched = False        

        total_log = np.zeros((1,2),dtype = np.float)
        average_qval_loss = policy_loss = pg_entropy = np.nan
        total_runs = 0        

        # to avoid wasting simulations
        min_fill_ratio = 0.95
        task_lst = [None for _ in self.sim_agents_discr]
        
        while True:
            # previous implementation (to be removed)
            #if self.memory_stored is not None and not qv_update_launched and not memory_fill_only:
            #    if not ray.get(self.nn_updater.hasMemoryPool.remote()): 
            #        raise('missing internal memory pool')
            
            # 1) start qv update based on existing memory stored
            if not memory_fill_only:
                if ray.get(self.nn_updater.hasMemoryPool.remote()) and not qv_update_launched:
                    # this check is required to ensure nn_udater has finished loading the memory 
                    #(in asynchronous setting this could take long)
                    updating_process = self.nn_updater.update_DQN_asynch.remote()
                    qv_update_launched = True
               
            # 2) start parallel simulations
            for i, agent in enumerate(self.sim_agents_discr):
                
                # 2.1) get info from complete processes
                if not ray.get(agent.isRunning.remote()): 
                    if task_lst[i] is not None:
                        partial_log, single_runs, successful_runs, pg_info = ray.get(task_lst[i])
                        
                        # implementation for A3C
                        if self.rl_mode == 'AC' and self.update_policy_online:
                            delta_theta_i, pg_loss, enthropy, valid_model = pg_info
                            if valid_model:
                                policy_loss += pg_loss
                                pg_entropy += enthropy
                                for k in self.model_pg.state_dict().keys():
                                    self.model_pg.state_dict()[k] += delta_theta_i[k]
                        
                        task_lst[i] = None
                        total_log = np.append(total_log, partial_log, axis = 0)
                        total_runs += single_runs
                        
                        self.shared_memory.addMemoryBatch(ray.get(agent.emptyLocalMemory.remote()) )
                        progress(round(1000*self.shared_memory.fill_ratio), 1000 , f'Fill memory - {self.shared_memory.size}')
                    
                    # 2.1.1) run another task if required
                    if not self.shared_memory.isPartiallyFull(min_fill_ratio):
                        
                        # check if model contains nan
                        """
                        for k,v in self.model_pg.state_dict().items():
                            if torch.isnan(v).any():
                                raise('nan in model')
                        """

                        agent.setAttribute.remote('model_pg',self.model_pg)
                        
                        # check model differences - REQUIRED to ensure synchronization before 
                        model_diff = 1
                        time_in = time.time()
                        while np.round(model_diff, 3) >= 0.001:
                            model_diff = self.model_pg.compare_weights( ray.get(agent.getAttributeValue.remote('model_pg')).state_dict()  )
                            if check_WhileTrue_timeout(time_in, 5):
                                raise('model in simulator different from current consensus model')
                        
                        task_lst[i] = agent.run.remote()
                    
            # if memory is (almost) full and task list empty, inform qv update it can stop
            if self.shared_memory.isPartiallyFull(min_fill_ratio) and all(tsk is None for tsk in task_lst):
                if qv_update_launched:
                    self.nn_updater.sentinel.remote()
                break
            
        print('')
        
        if self.rl_mode == 'AC' and self.update_policy_online:
            model_diff = np.round(self.model_pg.compare_weights(initial_pg_weights), 5)
            print(f'PG model change: {model_diff}')

            
        # log training history
        if qv_update_launched:
            
            # 1) compute qv average loss
            self.qval_loss_hist_iteration = np.array(ray.get(updating_process))
            average_qval_loss = np.round(np.average(self.qval_loss_hist_iteration), 3)
            print(f'QV. loss = {average_qval_loss}')
                  
            # 2) update policy and get policy loss (single batch, no need to average)) 
            if self.rl_mode == 'AC' and self.policy_loaded and not self.update_policy_online:
                model_update = ray.get(self.nn_updater.getAttributeValue.remote('model_pg'))
                model_diff = self.model_pg.compare_weights(model_update.cpu().state_dict().copy() )
                if model_diff > 0.001:
                    print(f'PG models MISMATCH: {model_diff}')
                    raise('update model different than SIM one')
                else:
                    policy_loss, pg_entropy = ray.get(self.nn_updater.update_DeepRL.remote('policy', self.shared_memory.memory ))
                
            #3) store computed losses
            self.get_log(total_runs, total_log, average_qval_loss, policy_loss , pg_entropy)


    ##################################################################################
    def get_log(self, total_runs, total_log, *args):
        
        #total log includes (for all runs): 1) steps since start 2) cumulative reward
        
        entry = np.zeros(len(args))
        
        for i,arg in enumerate(args):
            entry[i] = np.round(arg, 2)
        
        # total runs, steps since start, cumulative reward, average loss / total_successful_runs, entropy (pg training only)
        self.session_log = np.append( np.append( int(total_runs), np.round(np.average(total_log[1:,:], axis = 0),2)), entry)  

    
    ##################################################################################
    def validation_serialized(self):
        
        total_log = np.zeros((1,2),dtype = np.float)
        total_runs = 0
        tot_successful_runs = 0
            
        while not self.shared_memory.isFull():
            progress(round(1000*self.shared_memory.fill_ratio), 1000 , 'Fill memory')
            
            partial_log, single_runs, successful_runs, _ = self.sim_agents_discr[0].run_synch(use_NN = True)
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
                    partial_log, single_runs, successful_runs, _  = ray.get(agnt)
                    total_log = np.append(total_log, partial_log, axis = 0)
                    total_runs += single_runs
                    tot_successful_runs += successful_runs
            
            if not self.shared_memory.isPartiallyFull(min_fill_ratio):
                for agent in self.sim_agents_discr:
                    self.shared_memory.addMemoryBatch(ray.get(agent.emptyLocalMemory.remote()) )
                    progress(round(1000*self.shared_memory.fill_ratio), 1000 , f'Validation - {self.shared_memory.size}')
    
            if self.shared_memory.isPartiallyFull(min_fill_ratio): # or break_cycle:
                break
            
        print('')
        self.end_validation_routine(total_runs, tot_successful_runs, total_log)


    ##################################################################################
    def end_validation_routine(self, total_runs, tot_successful_runs, total_log):
        
        #total log includes (for all runs): 1) steps since start 2) cumulative reward
        self.get_log(total_runs, total_log, 0)
        if total_runs > 0 :
            success_ratio = np.round(tot_successful_runs/total_runs ,2)
        else:
            success_ratio = -1
        # session_log: total runs, steps since start, cumulative reward, average loss
        val_history = np.append(np.array([self.training_session_number, success_ratio ]) , self.session_log[:-1] )
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
    def runSequence(self, n_runs = 5, display_graphs = False, reset_optimizer = False):
        
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
                
            if i> 1 and not i % self.val_frequency:
                print('###################################################################')
                print('validation cycle start...')
                self.shared_memory.resetMemory(self.validation_set_size)
                if self.ray_parallelize:
                    self.validation_parallelized()
                else:
                    self.validation_serialized()
                print('end of validation cycle')
                print('###################################################################')
            """
            if i >= 30:
                losses = self.log_df[['av. q-val loss', 'av. policy loss']].to_numpy()
                if (losses[-1,:]>= 1e5*losses[-21]).any():
                    break
            """
        # save memory to reload for next iteration
        self.memory_stored.save(self.storage_path,self.net_name)


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
            
            """
            print('loading problem! did not find correct policy file!')
            self.check_model_version(print_out = True, mode = 'sim', save_v0 = True)
            if self.update_model_cpu():
                self.updateAgentsAttributesExcept()
            else:
                raise('Loading error')
            """

        
        if self.nn_updater is None:
            # if needed, initialize the model for the updating
            self.initialize_updater(reset_optimizer)
            if bool(self.memory_stored):
                # if memory stored is present, it means memory has been loaded
                self.load_memorystored_to_updater()

        else:
            # only update the net name (required to save/load net params)
            if self.ray_parallelize:
                self.nn_updater.setAttribute.remote('net_name',self.net_name)
            else:                
                self.nn_updater.setAttribute('net_name',self.net_name)
        
        
    ##################################################################################
    def initialize_updater(self, reset_optimizer = False):        
        # instanciate ModelUpdater actor -> everything here happens on GPU
        # initialize models (will be transferred to NN updater)
        model_pg = None
        model_qv = self.generate_model()
        if self.rl_mode == 'AC':
            model_pg = self.generate_model(pg_model=True)
        if self.ray_parallelize:
            self.nn_updater = Ray_RL_Updater.remote()
            self.updateRLUpdaterAttributesExcept()
            self.nn_updater.load_model.remote(model_qv, model_pg, reset_optimizer)
        else:
            self.nn_updater = RL_Updater()
            self.updateRLUpdaterAttributesExcept()
            self.nn_updater.load_model(model_qv, model_pg, reset_optimizer)
        
        
    ##################################################################################
    def q_val_bashplot(self):
        if self.bashplot and self.qval_loss_history is not None:       
            #print(f'history length= {self.loss_history_full.shape[0]}')
            if self.qval_loss_history.shape[0] > 2500:
                np.savetxt("loss_hist.csv", np.concatenate((np.arange(1,self.qval_loss_history[-2000:].shape[0]+1)[:,np.newaxis],self.qval_loss_history[-2000:,np.newaxis]), axis = 1), delimiter=",")
                plot_scatter(f = "loss_hist.csv",xs = None, ys = None, size = 20, colour = 'white',pch = '*', title = 'training loss')
        
        
    ##################################################################################
    def load_memorystored_to_updater(self):            
        if self.ray_parallelize:
            self.nn_updater.setAttribute.remote('memory_pool', self.memory_stored)
        else:
            self.nn_updater.setAttribute('memory_pool', self.memory_stored)

        
    ##################################################################################
    def end_training_round_routine(self):
        
        # update memory
        if self.memory_stored is None:
            self.memory_stored = self.shared_memory.cloneMemory()
            self.load_memorystored_to_updater()
            turnover_pctg_act = np.round(100*self.memory_stored.fill_ratio,2)
        else:
            new_memory = self.shared_memory.getMemoryAndEmpty()
            turnover_pctg_act = np.round(100*self.memory_stored.addMemoryBatch(new_memory),2)
            if self.ray_parallelize:
                self.nn_updater.update_memory_pool.remote(new_memory)
            else:
                self.nn_updater.update_memory_pool(new_memory)
            
        #print(f'memory size = {self.memory_stored.size*self.memory_stored.fill_ratio}')
        #print(f'effective memory turnover = {turnover_pctg_act}')
            
        # update log and history
        # (log structure: ['training session','epsilon', 'total single-run(s)','average single-run duration',\
        #    'average single-run reward' , 'average qval loss','average policy loss', 'N epochs', 'memory size', \
        #    action gen splits..., 'pg_entropy']  )
        if self.training_session_number > 1:
            new_row = [int(self.training_session_number), self.epsilon, self.ctrlr_probability , self.session_log[0] , self.session_log[1] ,\
                       self.session_log[2], self.session_log[3],self.session_log[4], self.n_epochs, self.replay_memory_size, \
                       self.splits[0], self.splits[1], self.splits[2] ,  self.session_log[5]   ]
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
        if self.ray_parallelize:
            # run simulations/training in parallel
            self.runParallelized()
        else:
            self.runSerialized()
            
        # we update the name in order to save it
        self.training_session_number +=1        
        self.update_net_name()
        
        self.end_training_round_routine()
       
        
    ##################################################################################
    # plot the training history
    def plot_training_log(self, init_epoch=0):
        #log_norm_df=(self.log_df-self.log_df.min())/(self.log_df.max()-self.log_df.min())
        #fig, ax = plt.subplots()
        # (log structure: 
        # 0: 'training session', 1: 'epsilon', 2: 'ctrl prob',3: 'total single-run(s)',
        # 4: 'average single-run duration', 5: 'average single-run reward' , 6: 'av. q-val loss', 
        # 7: 'av. policy loss','N epochs', 8: 'memory size', 9: 'split random/noisy',
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
        ax1_1.plot(self.log_df.iloc[init_epoch:][indicators[6]])
        #ax1_1.set_yscale('log')
        ax1_1.legend([indicators[6]])

        # 'av. policy loss'
        ax2_1 = fig_1.add_subplot(412)
        ax2_1.plot(self.log_df.iloc[init_epoch:][indicators[7]])
        ax2_1.legend([indicators[7]])
        
        # 'pg entropy'
        ax3_1 = fig_1.add_subplot(413)
        ax3_1.plot(self.log_df.iloc[init_epoch:][indicators[3]])
        ax3_1.legend([indicators[13]])

        # 'average single-run reward'
        ax4_1 = fig_1.add_subplot(414)
        ax4_1.plot(self.log_df.iloc[init_epoch:][indicators[5]])
        ax4_1.legend([indicators[5]])
   
    
        if self.update_policy_online:
            
            fig_2, ax2 = plt.subplots(2,1)

            ax2[0].plot(self.global_pg_loss)
            ax2[1].plot(self.global_pg_entropy)
            
            return fig, fig_1 , fig_2
    
        return fig, fig_1 

#%%

if __name__ == "__main__":
    #env = SimulationAgent(0, env = disc_GymStyle_Robot(n_bins_act=4), model = ConvModel())
    #rl_env = Multiprocess_RL_Environment('RobotEnv' , 'ConvModel')
    rl_env = Multiprocess_RL_Environment('CartPole' , 'LinearModel0')
    