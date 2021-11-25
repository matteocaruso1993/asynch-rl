# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:12:32 2020

@author: Enrico Regolin
"""

#%%
# This module contains the general framework of NN Class used for RL

import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


from ..rl.utilities import check_WhileTrue_timeout, check_saved
    
#%%


# NN class
class nnBase(nn.Module):
    """A base NN class comprising save/load , update and initializing methods """
    
    def __init__(self,model_version = -1, net_type='LinearModel0',lr =0.001, *argv, **kwargs):
        super().__init__()
        
        ## HERE WE PARSE THE ARGUMENTS
        self.model_version = model_version
    
        self.net_type = net_type
        self.lr = lr  

    ##########################################################################
    def count_NN_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    ##########################################################################
    def complete_initialization(self, kwargs):

        self.update_NN_structure()
        allowed_keys = {'softmax':False} 
                        
        # initialize all allowed keys
        self.__dict__.update((key, allowed_keys[key]) for key in allowed_keys)
        # and update the given keys by their given values
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        
        if self.softmax:
            self.sm = nn.Softmax(dim = 1)

        
        # define Adam optimizer
        self.initialize_optimizer() 
        
        # initialize mean squared error loss
        self.criterion_MSE = nn.MSELoss()
        
        # initialize random weights and zero gradient
        self.init_weights()
        self.init_gradient()


    ##########################################################################
    def initialize_optimizer(self):

        self.optimizer = optim.Adam(self.parameters(), lr= self.lr) #, weight_decay= self.weight_decay )
            #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma= self.gamma_scheduler)

    ##########################################################################
    def update_learning_rate(self, new_lr):
        self.lr = new_lr  
        self.initialize_optimizer()

    ##########################################################################
    ## HERE WE re-DEFINE THE LAYERS when needed        
    def update_NN_structure(self):
        pass

    ##########################################################################        
    def forward(self,x, **kwargs):
        return x 

    ##################################################################################
    def init_weights(self):
        if type(self) == nn.Conv1d or type(self) == nn.Conv2d or type(self) == nn.Linear:
            nn.init.uniform(self.weight, -0.01, 0.01)
            self.bias.data.fill_(0.01)
            
    ##################################################################################
    def init_gradient(self, device = torch.device('cpu') ):
        """ initializes net gradient to zero (to allow adding external gradients to it)"""
        generic_input = self.build_generic_input()
        if isinstance(generic_input, tuple):
            generic_input = tuple([g.to(device).float() for g in generic_input]) 
        else:
            generic_input =generic_input.to(device).float()
            
        net_output = self.forward(generic_input, return_map = True)
        if isinstance(net_output, tuple):
            loss_list = [torch.sum(out**2)  for out in net_output if out is not None]
            l = 0
            for ll in loss_list:
                l += ll 
        else:
            l = torch.sum(net_output**2)
        l.backward()
        self.optimizer.zero_grad()
        
        
    ##########################################################################        
    def build_generic_input(self):
        return torch.randn(self.get_net_input_shape())
        
        
    ##########################################################################        
    def get_net_input_shape(self):
        pass
    
    ##########################################################################
    # save the network parameters and the training history
    def save_net_params(self,path_log = None,net_name = 'my_net', epoch = 0, loss = 0):
        
        if path_log is None:
            path_log = os.path.dirname(os.path.abspath(__file__))
        
        filename_pt = os.path.join(path_log, net_name + '.pt')
        #opt_filename_pt = os.path.join(path_log, net_name + '_opt.pt')

        torch.save({
            'model_version': self.model_version,
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            },filename_pt)
        
        check_saved(filename_pt)
        
        
    ##########################################################################
    # load the network parameters and the training history
    def load_net_params(self,path_log,net_name, device, reset_optimizer = False):    
        
        filename_pt = os.path.join(path_log, net_name + '.pt')

        # this loop is required in parallelized mode, since it might take some 
        # time to successfully save the model params

        check_saved(filename_pt)
        
        try:
            checkpoint = torch.load(filename_pt, map_location=device)
            if 'model_version' in checkpoint:
                self.model_version = checkpoint['model_version']
            else: 
                parts = net_name.rsplit('_')
                self.model_version = int(parts[1])
            
        except Exception:
            
            print(f'Problems with net name {filename_pt}')
            
            with open(os.path.join(path_log,'train_log.txt'), 'r+') as f:
                logLines = f.readlines()
                if net_name +'\n' not in logLines:
                    print(f'model {net_name} not in list!!')
            
            print("*****************************************************")
            print("Can not load model parameters! Using previous version")
            print("*****************************************************")
            
            parts = net_name.rsplit('_')
            parts[1] = str(int(parts[1])-1)
            net_name__ = "_".join(parts)
               
            filename_pt_1 = os.path.join(path_log, net_name__ + '.pt')
            checkpoint = torch.load(filename_pt_1, map_location=device)
            self.model_version = checkpoint['model_version']+1
        
        epoch = checkpoint['epoch']
        model_state = checkpoint['model_state_dict']
        opt_state = checkpoint['optimizer_state_dict']
        loss = checkpoint['loss']

        self.init_layers(model_state)

        self.update_NN_structure()
        self.load_state_dict(model_state)
        self.to(device)  # required step when loading a model on a GPU (even if it was also trained on a GPU)
        
        self.initialize_optimizer()
        self.init_gradient(device)
        
        if not reset_optimizer:
            self.optimizer.load_state_dict(opt_state)
        else:
            print('Optimizer state has not been loaded!')
        
        self.eval()


    ##########################################################################
    # update net parameters based on state_dict() (which is loaded afterwards)
    def init_layers(self, model_state):
        pass
    

    ##########################################################################
    # compare weights before and after the update
    def compare_weights(self, state_dict_1, state_dict_0 = None):
                
        if state_dict_0 is None:
            state_dict_0 = self.state_dict()
        
        if identical_state_dicts(state_dict_0, state_dict_1):
            average_diff = 0.0
        else:
            diff_list = [torch.mean(torch.abs(v0[1] - v1[1])).item()  for v0,v1 in zip(state_dict_0.items(), state_dict_1.items())  ]
            average_diff = np.average(np.array(diff_list ))

        return average_diff


##########################################################################
def identical_state_dicts(state_dict_0, state_dict_1):
    if all([ torch.all(torch.eq(v0[1], v1[1])).item()  for v0,v1 in zip(state_dict_0.items(), state_dict_1.items()) ]):
        return True
    return False