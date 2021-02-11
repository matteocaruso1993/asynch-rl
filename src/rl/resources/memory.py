#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:12:02 2020

@author: Enrico Regolin
"""

import torch
import numpy as np
import random
import os

from rl.utilities import check_saved

#%%

class ReplayMemory():
    
    def __init__(self, size = 5000, minibatch_size= 32, update_only_with_full_memory = False, minibatch_multiple = 2 ):
        self.size = size
        self.minibatch_size = minibatch_size
        self.memory = []
        self.memory_full = False
        self.sampling_ready = False # will be made obsolete...
        self.fill_ratio = 0
        
        self.update_only_with_full_memory = update_only_with_full_memory
        self.minibatch_multiple = np.maximum(1,minibatch_multiple)  #current memory size has to be at least X times minibatch size

    # save transition to replay memory    
    def addInstance(self, memory_instance):
        #memory_instance = (state, action, reward, state_1, done)

        self.memory.append(memory_instance)
        pop_bool = self.update_memory_status()
        return pop_bool

        
    def update_memory_status(self):
        if not self.memory_full and len(self.memory) > self.size:
            self.memory_full = True

        # if update weights has not been yet allowed...
        if not self.sampling_ready:
            self.sampling_ready = self.memory_full
            if not self.update_only_with_full_memory and len(self.memory) > self.minibatch_multiple*self.minibatch_size:
                self.sampling_ready = True
                
        # if replay memory is full, remove the oldest transition
        if self.memory_full:
            self.memory.pop(0)
            self.fill_ratio = 1
            return True
        else:
            self.fill_ratio = len(self.memory) / self.size
            return False
            
    def extractMinibatch(self):
        # sample and unpack random minibatch 
        return unpack_batch(random.sample(self.memory, self.minibatch_size))
        
    def getActualSize(self):
        return len(self.memory)
   
    
    def save(self, path, net_name):
        filename = os.path.join(path, net_name + '_memory.pt')
        torch.save(self.memory,filename)
        check_saved(filename)


    def load(self, path, net_name):
        filename = os.path.join(path, net_name + '_memory.pt')
        self.memory = torch.load(filename)
        self.update_memory_status()

    
    def resetMemory(self, new_size = 0):
        if new_size > 0:
            self.size = new_size
        self.memory_full = False
        self.memory = []
        self.fill_ratio = 0

    
    def getMemoryAndEmpty(self):
        mem = self.memory
        self.resetMemory()
        return mem


    def readMemory(self):
        return self.memory

    
    def addMemoryBatch(self, token):
        replaced_items = 0
        for mem_instance in token:
            pop_bool = self.addInstance(mem_instance)
            replaced_items += pop_bool
        
        turnover_percentage = np.round(replaced_items/self.size , 3)
        
        return turnover_percentage
            

    def isPartiallyFull(self, ratio):
        return self.fill_ratio > ratio


    def isFull(self):
        return self.memory_full

    
    def cloneMemory(self, reset_existing = False):
        clonedObject = ReplayMemory(self.size , self.minibatch_size, self.update_only_with_full_memory , self.minibatch_multiple )
        
        if reset_existing:
            clonedObject.memory = self.getMemoryAndEmpty()
        else:
            clonedObject.memory = self.memory
            clonedObject.memory_full = self.memory_full
            clonedObject.sampling_ready = self.sampling_ready
            clonedObject.fill_ratio = self.fill_ratio

        return clonedObject


def unpack_batch(batch):
    
    states_0_tuple = tuple(element[0] for element in batch)
    states_1_tuple = tuple(element[3] for element in batch)
    
    if isinstance(states_0_tuple[0],tuple):
        states_0_batch_list = []
        states_1_batch_list = []
        for i in range(len(states_0_tuple[0])):
            states_0_batch_list.append( torch.cat(tuple( it[i] for it in states_0_tuple  )) )
            states_1_batch_list.append( torch.cat(tuple( it[i] for it in states_1_tuple  )) )
            
        state_batch   = tuple(states_0_batch_list)
        state_1_batch   = tuple(states_1_batch_list)
    else:
        state_batch   = torch.cat(states_0_tuple)
        state_1_batch   = torch.cat(states_1_tuple)
        
    action_batch  = torch.cat(tuple(element[1] for element in batch))
    reward_batch  = torch.cat(tuple(element[2] for element in batch))

    done_batch = (tuple(element[4] for element in batch) )
    action_idx_batch = (tuple(element[5] for element in batch) )

    return state_batch, action_batch, reward_batch, state_1_batch, done_batch, action_idx_batch