#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:49:10 2020

@author: Enrico Regolin
"""

#%%
import torch
from tqdm import tqdm
import random
import numpy as np

from copy import deepcopy
import os

from .memory import unpack_batch

from ..utilities import lineno


#%%

class RL_Updater():
    def __init__(self):
        # internal initializations
        self.memory_pool = None
        
        self.model_v = None

        # following attributes are in common with rl_env and will be updated externally
        self.net_name = None
        self.n_epochs = None
        self.move_to_cuda = None
        self.gamma   = None
        self.rl_mode = None
        self.storage_path      = None
        # only "initial" training session number is actually used inside RL_Updater
        self.training_session_number = None

        
    ##################################################################################        
    # required to check if model is inherited
    def update_memory_pool(self,new_batch):    
        self.memory_pool.addMemoryBatch(new_batch)         
        
        
    #################################################
    def save_model(self, path, model_name):

        for k,v in self.model_qv.state_dict().items():
            if torch.isnan(v).any():
                raise('not saving: nan tensor detected in model')

        self.model_qv.save_net_params(path, model_name)

        with open(os.path.join(path,'train_log.txt'), 'a+') as f:
            f.writelines(model_name + "\n")
            
        
        
    ##################################################################################        
    # required to check if model is inherited
    def save_updater_memory(self, path, name):
        self.memory_pool.save(path,name)
        
        
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
    # required since ray wrapper doesn't allow accessing attributes
    def hasMemoryPool(self):
        if self.memory_pool is not None:
            return self.memory_pool.fill_ratio > 0.8
        return False
       

    ##################################################################################        
    def move_models_to_cuda(self):
        self.model_qv = self.model_qv.cuda()
        if self.model_v is not None:
            self.model_v = self.model_v.cuda()
       
    #################################################
    # loading occurs based on current iteration loaded in the RL environment
    def load_model(self, model_qv, reset_optimizer = False):
        # first initialize device
        if self.move_to_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # load qv model
        self.model_qv = model_qv #self.rl_env.generate_model()
        if self.move_to_cuda:
            self.move_models_to_cuda()

        if self.training_session_number >= 1:
            self.model_qv.load_net_params(self.storage_path,self.net_name, self.device, reset_optimizer = reset_optimizer)

        for k,v in self.model_qv.state_dict().items():
            if torch.isnan(v).any():
                raise('nan tensor after load')

    
    #################################################
    def update_DeepRL(self):
        """synchronous update of DQL Network"""
        
        print(f'DQL Synchronous update started')
        total_loss = []

        for epoch in tqdm(range(self.n_epochs)):

            loss = self.qValue_loss_update(*self.memory_pool.extractMinibatch()[:-1])
            total_loss.append(loss.cpu().item())
            
        self.model_qv.model_version +=1

        return total_loss
        
    
    #################################################
    def qValue_loss_update(self, state_batch, action_batch, reward_batch, state_1_batch, done_batch):
        """ DQL update law """
        
        
        if self.move_to_cuda:  # put on GPU if CUDA is available
            if isinstance(state_batch,tuple):
                state_batch   = tuple([s.cuda() for s in state_batch])
                state_1_batch = tuple([s.cuda() for s in state_1_batch])
            else:
                state_batch = state_batch.cuda()
                state_1_batch = state_1_batch.cuda()
            
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            

        # get output for the next state
        with torch.no_grad():
            output_1_batch = self.model_qv(state_1_batch)
        
            
        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if done_batch[i]  #minibatch[i][4]
                                  else reward_batch[i] + self.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(reward_batch))))
        
        
        # extract Q-value
        # calculates Q value corresponding to all actions, then selects those corresponding to the actions taken
        self.model_qv.optimizer.zero_grad()
        q_vals = self.model_qv(state_batch)
        
        q_value = torch.sum( q_vals * action_batch, dim=1)
        y_batch = y_batch.detach()
        loss_qval = self.model_qv.criterion_MSE(q_value, y_batch)
        
        loss_qval.backward()
        
        for name, param in self.model_qv.named_parameters():
            FINITE_GRAD = torch.isfinite(param.grad).all() 
            if not FINITE_GRAD:
                print(name, param.grad)
        
        self.model_qv.optimizer.step()  
        
        for k,v in self.model_qv.state_dict().items():
            if torch.isnan(v).any():
                raise('nan tensor in model after update')
        
        
        return loss_qval


def extract_tensor_batch(t, batch_size):
    """ batch extraction from tensor """
    
    # extracs batch from first dimension (only works for 2D tensors)
    idx = torch.randperm(t.nelement())
    return t.view(-1)[idx][:batch_size].view(batch_size,1)
    