# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 16:12:32 2020

@author: Enrico Regolin
"""

#%%
# This module contains the general framework of NN Class used for RL

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
    

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
    def complete_initialization(self, kwargs):

        self.update_NN_structure()
        
        """
        ## HERE WE DEAL WITH ADDITIONAL (OPTIONAL) ARGUMENTS
        # lr = optimizer learning rate
        default_optimizer = optim.Adam(self.parameters(),self.lr)
        default_loss_function = nn.MSELoss()  #this might be made changable in the future 
        default_weights_dict = {new_list: .1 for new_list in range(1,11)} 
        """
        
        allowed_keys = {'softmax':True, 'conv_no_grad':False} #,'BATCH_SIZE':200,'device': torch.device("cpu"), \
                        #'optimizer' : default_optimizer , 'loss_function' : default_loss_function, \
                        #'weights_dict' : default_weights_dict }#, 'VAL_PCT':0.25 , \
                        
        # initialize all allowed keys
        self.__dict__.update((key, allowed_keys[key]) for key in allowed_keys)
        # and update the given keys by their given values
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        
        # define Adam optimizer
        self.initialize_optimizer() 
        
        # initialize mean squared error loss
        self.criterion_MSE = nn.MSELoss()


    ##########################################################################
    def initialize_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr= self.lr)

    ##########################################################################
    def update_learning_rate(self, new_lr):
        self.lr = new_lr  
        self.initialize_optimizer()

    ##########################################################################
    ## HERE WE re-DEFINE THE LAYERS when needed        
    def update_NN_structure(self):
        pass

    ##########################################################################        
    def forward(self,x):
        return x 

    ##################################################################################
    def init_weights(self):
        if type(self) == nn.Conv1d or type(self) == nn.Conv2d or type(self) == nn.Linear:
            nn.init.uniform(self.weight, -0.01, 0.01)
            self.bias.data.fill_(0.01)
    
    
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
        
        
    ##########################################################################
    # load the network parameters and the training history
    def load_net_params(self,path_log,net_name, device, reset_optimizer = False):    
        
        filename_pt = os.path.join(path_log, net_name + '.pt')

        # this loop is required in parallelized mode, since it might take some 
        # time to successfully save the model params
        count = 0
        while count < 30:
            try:
                checkpoint = torch.load(filename_pt, map_location=device)
                break
            except Exception:
                time.sleep(1)
                count +=1
                print(f'tentative : {count}')
        
        self.model_version = checkpoint['model_version']
        epoch = checkpoint['epoch']
        model_state = checkpoint['model_state_dict']
        opt_state = checkpoint['optimizer_state_dict']
        loss = checkpoint['loss']

        self.init_layers(model_state)

        self.update_NN_structure()
        self.load_state_dict(model_state)
        self.to(device)  # required step when loading a model on a GPU (even if it was also trained on a GPU)
        
        self.initialize_optimizer()
        
        if not reset_optimizer:
            self.optimizer.load_state_dict(opt_state)
        else:
            print('Optimizer state has not been loaded!')
        
        self.eval()


    ##########################################################################
    # update net parameters based on state_dict() (which is loaded afterwards)
    def init_layers(self, model_state):
        pass