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
#import time

from copy import deepcopy
import os

from .memory import unpack_batch

from rl.utilities import lineno


#%%

class RL_Updater():
    def __init__(self):
        
        # internal initializations
        self.PG_update_failed_once = False
        self.memory_pool = None
        #self.nn_updating = False

        # these will have to be initialized from RL_env
        self.beta_PG       = None # 0.01
        self.n_epochs_PG   = None # 100
        self.batch_size_PG = None # 32

        # following attributes are in common with rl_env and will be updated externally
        self.net_name = None
        self.n_epochs = None
        self.move_to_cuda = None
        self.gamma   = None
        self.rl_mode = None
        self.pg_partial_update = None
        self.storage_path      = None
        # only "initial" training session number is actually used inside RL_Updater
        self.training_session_number = None

        
    ##################################################################################        
    # required to check if model is inherited
    def update_memory_pool(self,new_batch):    
        self.memory_pool.addMemoryBatch(new_batch)         
        
    #################################################
    def save_model(self, path, model_name):
        self.model_qv.save_net_params(path, model_name)
        if self.rl_mode == 'AC':
            self.model_pg.save_net_params(path, model_name+'_policy')
        
        with open(os.path.join(path,'train_log.txt'), 'a+') as f:
                f.writelines(model_name + "\n")
                if self.rl_mode == 'AC':
                    f.writelines(model_name+'_policy'+ "\n")
        
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
        return self.memory_pool is not None
       
    #################################################
    # loading occurs based on current iteration loaded in the RL environment
    def load_model(self, model_qv, model_pg=None, reset_optimizer = False):
        # first initialize device
        if self.move_to_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # load qv model
        self.model_qv = model_qv #self.rl_env.generate_model()
        if self.move_to_cuda:
            self.model_qv = self.model_qv.cuda()

        if self.training_session_number >= 1:
            self.model_qv.load_net_params(self.storage_path,self.net_name, self.device, reset_optimizer = reset_optimizer)
        else:
            self.model_qv.init_weights()

        # load pg model
        if self.rl_mode == 'AC':
            self.model_pg = model_pg #self.rl_env.generate_model(pg_model=True)
            if self.move_to_cuda:
                self.model_pg = self.model_pg.cuda()
            try:
                self.model_pg.load_net_params(self.storage_path,self.net_name+'_policy', self.device, reset_optimizer = reset_optimizer)
            except Exception:
                raise('Existing PG model not found!')
    
    
    #################################################
    def update_DeepRL(self, net = 'state_value', policy_memory = None):
        """synchronous update of Reinforcement Learning Deep Network"""
        
        #self.nn_updating = True
        print(f'Synchronous update started: {net}')
        total_loss = []
        total_mismatch = 0
            
        if net == 'state_value':

            for epoch in tqdm(range(self.n_epochs)):

                loss = self.qValue_loss_update(*self.memory_pool.extractMinibatch()[:-1])
                total_loss.append(loss.cpu().detach().numpy())
                
            self.model_qv.model_version +=1
            pg_entropy = None

        elif net == 'policy' and policy_memory is not None :
            
            state_dict_0 = deepcopy(self.model_pg.state_dict()) #.copy()
                        
            total_loss, n_mismatch, pg_entropy = self.policy_loss_update(*tuple([batch for i, batch in enumerate(unpack_batch(policy_memory)) if i in [0,1,2,3,5]]))
            
            invalid_samples_pctg =  np.round(100*n_mismatch/ len(policy_memory), 2)
            print(f'Update finished. mismatch % =  {invalid_samples_pctg}%') 
            
            if self.pg_partial_update:
                self.model_pg.load_conv_params(self.storage_path,self.net_name, self.device)
            
            update_diff = self.model_pg.compare_weights(state_dict_0)
            print(f'PG. average loss per epoch : {round(total_loss,3)}, model update = {round(update_diff,8)}')
            print(f'sample entropy : {round(pg_entropy,3)}')
            
            #print(state_dict_0['fc1.weight'][:5,:])
            #print(self.model_pg.fc1.weight[:5,:])
            
            self.model_pg.model_version +=1

        else:
            raise('Undefined Net-type')
        
        #self.nn_updating = False
        return total_loss, pg_entropy
    

    
    #################################################
    def policy_loss_update(self, state_batch, action_batch, reward_batch, state_1_batch, actual_idxs_batch):
        
        print('Actions Distribution')
        a = np.array(actual_idxs_batch)
        unique, counts = np.unique(a, return_counts=True)
        print(dict(zip(unique, counts)))
        
        
        if self.move_to_cuda:  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()
        
        # we re-compute the probabilities, this time in batch (they have to match with the outputs obtained during the simulation)
        prob_distribs_batch = self.model_pg(state_batch.float())   
        #action_idx_batch = torch.argmax(prob_distribs_batch, dim = 1)
        action_idx_batch = actual_idxs_batch
        
        action_batch_grad = torch.zeros(action_batch.shape).cuda()
        action_batch_grad[torch.arange(action_batch_grad.size(0)),action_idx_batch] = 1
            
        prob_action_batch = prob_distribs_batch[torch.arange(prob_distribs_batch.size(0)), action_idx_batch].unsqueeze(1)
        entropy = -torch.sum(prob_distribs_batch*torch.log(prob_distribs_batch),dim = 1).unsqueeze(1)
        
        with torch.no_grad():
            advantage = reward_batch + (self.gamma *torch.max(self.model_qv(state_1_batch.float()), dim = 1, keepdim = True)[0]\
                - torch.max(self.model_qv(state_batch.float()), dim = 1, keepdim = True)[0])

        if (action_batch == action_batch_grad).all().item():
            loss_vector =  -torch.log(prob_action_batch)*advantage + self.beta_PG*entropy 
            #loss_policy = 1000*self.model_pg.criterion_MSE(prob_action_batch, advantage)
            n_invalid = 0
            self.PG_update_failed_once = False
            
            #print(f'pg_loss = {torch.mean(-torch.log(prob_action_batch)*advantage)}')
            #print(f'entropy = {torch.mean(self.beta_PG*entropy)}')
        else:
            print('WARNING: selected action mismatch detected')
            valid_rows   = (action_batch == action_batch_grad).all(dim = 1)
            invalid_rows = (action_batch != action_batch_grad).any(dim = 1)

            """ # this was a test to "verify" that mismatch is compatible with a numerical error
            # max prob index            
            print(prob_distribs_batch[invalid_rows]) #  , torch.arange(prob_distribs_batch.size(1)) ]
            print( torch.argmax(prob_distribs_batch[invalid_rows], dim = 1) )
            # actual action index
            print( torch.argmax(action_batch[invalid_rows], dim = 1) )
            """
            
            loss_vector =  -torch.log(prob_action_batch[valid_rows])*advantage[valid_rows] + self.beta_PG*entropy[valid_rows]
            n_invalid = torch.sum(invalid_rows).item()
            
            pctg_mismatch = round( 100*invalid_rows.nonzero().squeeze(1).shape[0] / loss_vector.shape[0], 2 )
            print(f'Mismatch pctg : {pctg_mismatch}!')
            
            if pctg_mismatch > 10:
                
                if self.PG_update_failed_once:
                    raise('mismatch problem')
                else:
                    self.PG_update_failed_once = True
 
        if self.PG_update_failed_once:
            loss_policy = torch.tensor(float('nan'))
        else:
            if self.n_epochs_PG < 10:
                self.model_pg.optimizer.zero_grad()
                loss_policy = torch.mean(loss_vector)
                loss_policy.backward()
                self.model_pg.optimizer.step()  
            else:
                losses = []
                batch_size = min(self.batch_size_PG, round(0.5*loss_vector.shape[0]))
                self.model_pg.optimizer.zero_grad()
                
                for _ in tqdm(range(self.n_epochs_PG)):
                    #self.model_pg.optimizer.zero_grad()
                    loss = torch.mean(extract_tensor_batch(loss_vector, batch_size))
                    loss.backward(retain_graph = True)
                    losses.append(loss.unsqueeze(0))
                    #self.model_pg.optimizer.step() 
                # last run erases the gradient
                #self.model_pg.optimizer.zero_grad()
                loss = torch.mean(extract_tensor_batch(loss_vector, batch_size))
                loss.backward()
                losses.append(loss.unsqueeze(0))
                self.model_pg.optimizer.step()  
                
                loss_policy = torch.mean(torch.cat(losses))
        
        return loss_policy.item(), n_invalid, torch.mean(entropy).item()
    
    
    #################################################
    def qValue_loss_update(self, state_batch, action_batch, reward_batch, state_1_batch, done_batch):
        
        if self.move_to_cuda:  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # get output for the next state
        with torch.no_grad():
            output_1_batch = self.model_qv(state_1_batch.float())
        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if done_batch[i]  #minibatch[i][4]
                                  else reward_batch[i] + self.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(reward_batch))))
        # extract Q-value
        # calculates Q value corresponding to all actions, then selects those corresponding to the actions taken
        q_value = torch.sum(self.model_qv(state_batch.float()) * action_batch, dim=1)
        
        self.model_qv.optimizer.zero_grad()
        y_batch = y_batch.detach()
        loss_qval = self.model_qv.criterion_MSE(q_value, y_batch)
        loss_qval.backward()
        self.model_qv.optimizer.step()  
        
        return loss_qval


def extract_tensor_batch(t, batch_size):
    # extracs batch from first dimension (only works for 2D tensors)
    idx = torch.randperm(t.nelement())
    return t.view(-1)[idx][:batch_size].view(batch_size,1)
