# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 12:22:12 2020

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
#
# NN class

class LinearModel(nnBase):
    def __init__(self,model_version = -1, net_type='LinearModel0',lr =0.001, n_outputs=1, n_inputs=10,*argv,**kwargs):
        super().__init__(model_version, net_type, lr)
    
    
        self.net_depth = max(1,len(argv))
        self.n_inputs = n_inputs
        
        if not argv:
            self.n_layer_1 = 1
        else:
            for i,arg in zip(range(1,self.net_depth+1),argv):
                setattr(self, 'n_layer_'+str(i), arg)
        self.n_actions = n_outputs
    
    
        self.update_NN_structure()    
    
    ## let's make the net dynamic by using the "property" decorator
        @property
        def net_depth(self):
            return self._net_depth
        @net_depth.setter
        def net_depth(self,value):
            if value > 5:
                raise AttributeError(f'NN is too deep')
            else:
                self._net_depth = value     
        
        @property
        def n_inputs(self):
            return self._n_inputs
        @n_inputs.setter
        def n_inputs(self,value):
            self._n_inputs = value
            
        @property
        def n_layer_1(self):
            return self._n_layer_1
        @n_layer_1.setter
        def n_layer_1(self,value):
            self._n_layer_1 = value
             
        @property
        def n_layer_2(self):
            return self._n_layer_2
        @n_layer_2.setter
        def n_layer_2(self,value):
            self._n_layer_2 = value     
        
        @property
        def n_layer_3(self):
            return self._n_layer_3
        @n_layer_3.setter
        def n_layer_3(self,value):
            self._n_layer_3 = value     

        @property
        def n_layer_4(self):
            return self._n_layer_4
        @n_layer_4.setter
        def n_layer_4(self,value):
            self._n_layer_4 = value     

        @property
        def n_layer_5(self):
            return self._n_layer_5
        @n_layer_5.setter
        def n_layer_5(self,value):
            self._n_layer_5 = value     

        @property
        def n_outputs(self):
            return self._n_outputs
        @n_outputs.setter
        def n_outputs(self,value):
            self._n_outputs = value     

        @property
        def fc1(self):
            return self._fc1
        @fc1.setter
        def fc1(self,value):
            self._fc1 = value
        
        @property
        def fc2(self):
            return self._fc2
        @fc2.setter
        def fc2(self,value):
            self._fc2 = value
            
        @property
        def fc3(self):
            return self._fc3
        @fc3.setter
        def fc3(self,value):
            self._fc3 = value
        
        @property
        def fc4(self):
            return self._fc4
        @fc4.setter
        def fc4(self,value):
            self._fc4 = value
        
        @property
        def fc5(self):
            return self._fc5
        @fc5.setter
        def fc5(self,value):
            self._fc5 = value
        
        @property
        def fc_output(self):
            return self._fc_output
        @fc_output.setter
        def fc_output(self,value):
            self._fc_output = value

        
        # this should always be run in the child classes after initialization
        self.complete_initialization(kwargs)
    
    
    ##########################################################################        
    def get_net_input_shape(self):
        return (1, 1 , self.n_inputs )

    
    ##########################################################################
    ## HERE WE re-DEFINE THE LAYERS when needed        
    def update_NN_structure(self):
        # defining first layer
        self.fc1 = nn.Linear(self.n_inputs, self.n_layer_1)
        # defining layers 2,3,etc.
        for i in range(1,self.net_depth):
            setattr(self, 'fc'+str(i+1), nn.Linear(getattr(self, 'n_layer_'+str(i)),getattr(self, 'n_layer_'+str(i+1) ) ) )
        # define last layer
        last_layer_width = getattr(self, 'n_layer_'+str(self.net_depth)  )
        self.fc_output = nn.Linear(last_layer_width, self.n_actions)

    ##########################################################################        
    ## HERE WE DEFINE THE PATH
    # inside the forward function we can create CONDITIONAL PATHS FOR CERTAIN LAYERS!!
    def forward(self,x, **kwargs):
        iterate_idx = 1
        for attr in self._modules:
            if 'fc'+str(iterate_idx) in attr:
                x = F.relu(self._modules[attr](x))
                iterate_idx += 1
                if iterate_idx > self.net_depth:
                    break
        x = self.fc_output(x)
        if self.softmax:
            #sm = nn.Softmax(dim = 1)
            x = self.sm(x)
            
        if 'return_map' in kwargs:
            return x,None
        else:
            return x #F.log_softmax(x,dim=1)  # we return the log softmax (sum of probabilities across the classes = 1)
    

    ##########################################################################
    # update net parameters based on state_dict() (which is loaded afterwards)
    def init_layers(self, model_state):         
    
        self.n_inputs =  (model_state['fc1.weight']).shape[1]
        self.net_depth = int(len(model_state)/2)-1
        for i in range(1,self.net_depth+1):
            setattr(self, 'n_layer_'+str(i), (model_state['fc' + str(i) + '.weight']).shape[0])
        self.n_actions =  (model_state['fc_output.weight']).shape[0]


#%%

# item size = 1xN (periods are considered as channels)
# item size is not needed as input for the conv modules (different from linear layers)

#kernel_size_in = 3 or 5
#strides   = [5,3] --> strides are used in maxpool layers
#C1/C2:  number of channels out of first/second convolution (arbitrary)
# F1/F2: outputs of linear layers
"""
class ConvModel(nnBase):
    # batch size is not an input to conv layers definition
    # make sure N%stride = 0 for all elements in strides
    def __init__(self, model_version = -1, net_type='ConvModel0',lr =1e-6,  n_actions = 2, channels_in = 4, N_in = [165 , 4], \
                 C1 = 16, C2 = 3, C1_robot = 10, C2_robot = 4 ,F1 = 60, F2 = 60, F3 = 20 ,kernel_size_in=5, \
                     strides=[5,3],**kwargs):
        
        #ConvModel, self
        super().__init__(model_version, net_type, lr)
        if N_in[0] % (strides[0]*strides[1]):
            raise ValueError('N in and strides not aligned')
            
        #self.model_version = model_version
        #self.net_type = net_type
        #self.lr = lr
        
        self.N_in = N_in        
        self.channels_in = channels_in
        self.n_actions = n_actions
        self.C1 = C1
        self.C2 = C2
        self.C1_robot = C1_robot
        self.C2_robot = C2_robot
        self.F1 = F1
        self.F2 = F2
        self.F3 = F3
        self.kernel_size_in = kernel_size_in
        self.strides = strides
        
        # in AC mode, following weights are updated separately
        self.independent_weights = ['fc1.weight', 'fc1.bias','fc2.weight', 'fc2.bias','fc3.weight', 'fc3.bias','fc_output.weight', 'fc_output.bias']

        # this should always be run in the child classes after initialization
        self.complete_initialization(kwargs)
        
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'total NN trainable parameters: {pytorch_total_params}')

        

    ##########################################################################        
    def get_net_input_shape(self):
        return (1, self.channels_in, sum(self.N_in) )

        
    ##########################################################################
    def update_NN_structure(self):
        ## LIDAR RAYS CONVOLUTION
        # input [B, channels_in , N_in[0]]
        self.conv1_lidar = nn.Conv1d(in_channels=self.channels_in, out_channels=self.C1, kernel_size=self.kernel_size_in, padding=round((self.kernel_size_in-1)/2)  ) 
        # padding allows N not to be reduced
        # [B, C1, N_in[0]]
        self.maxpool1_lidar = nn.MaxPool1d(kernel_size=7, stride=self.strides[0], padding=3) # stride == 5
        # [B, C1, N_in[0]/strides[0]]    (no alignment issue if N is a multiple of strides[0] )
        self.conv2_lidar = nn.Conv1d(self.C1, self.C2, kernel_size=3, padding=1)
        # [B, C2, N_in[0]/strides[0]] #only #channels change from convolution
        self.maxpool2_lidar = nn.MaxPool1d(kernel_size=5, stride=self.strides[1], padding = 2) # stride == 3
        # [B, C2, N_postconv = N/(strides[0]*strides[1])] #only #channels change from convolution
       
        ## ROBOT MOVEMENT CONVOLUTION
        # input [B, channels_in , N_in[1]]
        self.conv1_robot = nn.Conv1d(in_channels=self.channels_in, out_channels=self.C1_robot, kernel_size=3, padding=1) 
        # padding allows N not to be reduced
        # [B, C1_robot, N_in[1]]
        self.conv2_robot = nn.Conv1d(self.C1_robot, self.C2_robot, kernel_size=3, padding=1)
        # [B, C2_robot, N_in[1]] #only #channels change from convolution
       
        #N_postconv_lidar =   # 165/(3*5) = 11
        N_linear_in = round(self.N_in[0]/(self.strides[0]*self.strides[1])*self.C2 + self.N_in[1]*self.C2_robot)
        
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
        x_lidar = x[:,:,:self.N_in[0]]
        x_robot = x[:,:,-self.N_in[1]:]
        
        x_lidar = F.relu(self.conv1_lidar(x_lidar))
        x_lidar = self.maxpool1_lidar(x_lidar)

        x_lidar = F.relu(self.conv2_lidar(x_lidar))
        x_lidar = self.maxpool2_lidar(x_lidar)
        
        x_robot = F.relu(self.conv1_robot(x_robot))
        x_robot = F.relu(self.conv2_robot(x_robot))

        x = torch.cat((x_lidar.flatten(1), x_robot.flatten(1)),dim = 1) # flatten the tensor starting at dimension 1 (to keep batches intact?)    
   
        return x
    
    
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
        
        self.channels_in = model_state['conv1_lidar.weight'].shape[1]
        self.n_actions = model_state['fc_output.weight'].shape[0]
        self.C1 = model_state['conv1_lidar.weight'].shape[0]  #16
        self.C2 = model_state['conv2_lidar.weight'].shape[0]  #16
        self.C1_robot = model_state['conv1_robot.weight'].shape[0]
        self.C2_robot = model_state['conv2_robot.weight'].shape[0]
        self.F1 = model_state['fc1.weight'].shape[0]
        self.F2 = model_state['fc2.weight'].shape[0]
        self.F3 = model_state['fc3.weight'].shape[0]
        self.kernel_size_in = model_state['conv1_lidar.weight'].shape[2]
        
        N_linear_in = round(self.N_in[0]/(self.strides[0]*self.strides[1])*self.C2 + self.N_in[1]*self.C2_robot)
        if N_linear_in != model_state['fc1.weight'].shape[1]:
            raise("NN consistency error")


    ##########################################################################
    # compare weights before and after the update
    def compare_weights(self, state_dict_1, state_dict_0 = None):
        
        average_diff = super().compare_weights(state_dict_1, state_dict_0)
                
        return average_diff
"""
    
#%%
#test

#model = ConvModel('ConvModel', n_actions = 9 )


#test_tensor = torch.rand(32,4,165)
#out = model(test_tensor)