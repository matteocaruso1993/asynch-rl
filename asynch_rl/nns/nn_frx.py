#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 14:09:16 2021

@author: Enrico Regolin
"""

#%%
import os

import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_nn import nnBase
   

#%%

# item size = 1xN (periods are considered as channels)
# item size is not needed as input for the conv modules (different from linear layers)

#kernel_size_in = 3 or 5
#strides   = [5,3] --> strides are used in maxpool layers
#C1/C2:  number of channels out of first/second convolution (arbitrary)
# F1/F2: outputs of linear layers

#%%

class NN_frx(nnBase):
    # batch size is not an input to conv layers definition
    # make sure N%stride = 0 for all elements in strides
    def __init__(self, model_version = -1, net_type='ConvModel0',lr =1e-6,  n_actions = 21, channels_in = 4, N_in = [240,2] , \
                 C1 = 8, C2 = 3, F1 = 50, F2 = 50, F3 = 20 ,kernel_size_in=5, \
                     strides=[5,3],**kwargs):
        
        #ConvModel, self
        super().__init__(model_version, net_type, lr)
        if N_in[0] % (strides[0]*strides[1]):
            raise ValueError('N in and strides not aligned')
            
        
        self.N_in = N_in        
        self.channels_in = channels_in
        self.n_actions = n_actions
        self.C1 = C1
        self.C2 = C2
        self.F1 = F1
        self.F2 = F2
        self.F3 = F3
        self.kernel_size_in = kernel_size_in
        self.strides = strides
        
        # in AC mode, following weights are updated separately
        self.independent_weights = ['fc1.weight', 'fc1.bias','fc2.weight', 'fc2.bias','fc3.weight', 'fc3.bias','fc_output.weight', 'fc_output.bias']

        # this should always be run in the child classes after initialization
        self.complete_initialization(kwargs)
        
        
    ##########################################################################        
    def build_generic_input(self):
        a,b = self.get_net_input_shape()
        return torch.randn(a), torch.randn(b)


    ##########################################################################        
    def get_net_input_shape(self):
        return (1, self.channels_in, self.N_in[0] ), (1, self.N_in[1])

        
    ##########################################################################
    def update_NN_structure(self):
        ## LIDAR RAYS CONVOLUTION
        # input [B, channels_in , N_in[0]]
        self.conv1 = nn.Conv1d(in_channels= self.channels_in, out_channels=self.C1, kernel_size=self.kernel_size_in, padding=round((self.kernel_size_in-1)/2)  ) 
        # padding allows N not to be reduced
        # [B, C1, N_in[0]]
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=self.strides[0]) # stride == 5
        # [B, C1, N_in[0]/strides[0]]    (no alignment issue if N is a multiple of strides[0] )
        self.conv2 = nn.Conv1d(self.C1, self.C2, kernel_size=3, padding=1)
        # [B, C2, N_in[0]/strides[0]] #only #channels change from convolution
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=self.strides[1]) # stride == 3
        # [B, C2, N_postconv = N/(strides[0]*strides[1])] #only #channels change from convolution
       
        #N_postconv_lidar =   # 165/(3*5) = 11
        N_linear_in = round(self.N_in[0]/(self.strides[0]*self.strides[1])*self.C2 + self.N_in[1])
        
        # (final) fully connected layers
        self.fc1 = nn.Linear(N_linear_in, self.F1)
        # [B, F1]
        self.fc2 = nn.Linear(self.F1, self.F2)
        # [B, F2]
        self.fc3 = nn.Linear(self.F2, self.F3)
        # [B, 2]
        self.fc_output = nn.Linear(self.F3, self.n_actions)

    ##########################################################################
    def forward(self,x):

        x = self.conv_forward(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc_output(x)

        if self.softmax:
            sm = nn.Softmax(dim = 1)
            x = sm(x)

        return x
    
    
    ##########################################################################
    def conv_forward(self,x):
        """ takes a tuple of two tensors as input """
        
        x1 = x[0]
        
        x1 = F.relu(self.conv1(x1))
        x1 = self.maxpool1(x1)
        x1 = F.relu(self.conv2(x1))
        x1 = self.maxpool2(x1)
   
        return torch.cat((x1.flatten(1), x[1].flatten(1)),dim = 1)
    
    
    ##########################################################################
    # load network parameters of only convolutional layers (net structure is assumed correct!)
    def load_conv_params(self,path_log,net_name, device):    
        
        filename_pt = os.path.join(path_log, net_name + '.pt')
        checkpoint = torch.load(filename_pt, map_location=device)
        pretrained_dict = checkpoint['model_state_dict']

        model_dict = self.state_dict(self.independent_weights)
        
        # 1. filter out unnecessary keys
        pretrained_dict = {k : v for k, v in pretrained_dict.items() if k not in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.load_state_dict(pretrained_dict)

        
        self.to(device)  # required step when loading a model on a GPU (even if it was also trained on a GPU)
        self.eval()
    
    ##########################################################################
    # update net parameters based on state_dict() (which is loaded afterwards)
    def init_layers(self, model_state):         
        
        self.channels_in = model_state['conv1.weight'].shape[1]
        self.n_actions = model_state['fc_output.weight'].shape[0]
        self.C1 = model_state['conv1.weight'].shape[0]  #16
        self.C2 = model_state['conv2.weight'].shape[0]  #16
        self.F1 = model_state['fc1.weight'].shape[0]
        self.F2 = model_state['fc2.weight'].shape[0]
        self.F3 = model_state['fc3.weight'].shape[0]
        self.kernel_size_in = model_state['conv1.weight'].shape[2]
        
        N_linear_in = round(self.N_in[0]/(self.strides[0]*self.strides[1])*self.C2 + self.N_in[1])
        if N_linear_in != model_state['fc1.weight'].shape[1]:
            raise("NN consistency error")

    ##########################################################################
    # compare weights before and after the update
    def compare_weights(self, state_dict_1, state_dict_0 = None):
        
        average_diff = super().compare_weights(state_dict_1, state_dict_0)
            
        return average_diff


    
#%%
#test

if __name__ == "__main__":
    model = NN_forex('ConvModel', n_actions = 21 )
    
    
    test_tensor = torch.rand(32,6,240)
    
    out = model(test_tensor)