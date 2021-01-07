# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:51:55 2021

@author: enric
"""

# 4 straight
import numpy as np
import torch
import gym
from gym import spaces

import colorama
from colorama import Back, Style
colorama.init()


#%%

class ConnectFour(gym.Env):
    
    def __init__(self, rewards = [100,1], pause = 0, print_out = False, use_NN = False):
        
        self.env_type = 'Connect4'
        
        self.width = 7
        self.height = 6
        self.max_n_moves = self.width * self.height
        
        # externally editable params
        self.print_out = print_out
        self.pause = 0
        if self.print_out:
            self.pause = pause
        self.use_NN = use_NN
        self.rewards = rewards

        self.action_space = spaces.Box(low=np.array([0]), high=np.array([self.width-1])) 
        
        self.board = np.zeros((self.height+1,self.width), dtype = np.int8) 
        self.next_mover = None
    
    
    ###################################
    """ these three methods are needed in sim env / rl env"""
    def get_max_iterations(self):
        return int(self.max_n_moves)
 
    def get_actions_structure(self):
        pass # implemented in wrapper

    def get_observations_structure(self):
        return len(self.get_state())
    ###################################

        
    def update_model(self, external_model):
        self.model = external_model

    
    def render(self, highlight_coords = []):

        print('')
        print('')
        #print(np.flipud(self.board))
        for i,row in enumerate(np.flipud(self.board)):
            for j,elem in enumerate(row):
                endcond = '\n' if j == self.width-1 else " "
                if (self.height-1-i,j) in  highlight_coords:
                    print(Back.RED   + sym_rep(elem) + Style.RESET_ALL, end=endcond, flush=True)
                    #print('4', end=endcond, flush=True)
                else:
                    print(sym_rep(elem), end=endcond, flush=True)
                if i == 0 and j == self.width-1:
                    print('--------------')
        print('--------------')
    
    def reset(self, **kwargs):
        
        self.moves_counter = 0
        
        self.board = np.zeros((self.height+1,self.width), dtype = np.int8)         
        if np.random.random()<0.5:
            self.next_mover = 'player'
        
        self.next_mover = 'opponent'
        opp_move = self.generate_random_move()
        self.apply_move(opp_move, own = False)
        self.next_mover = 'player'
        
    def apply_move(self, action, own = True):
        free_slot = np.where(self.board[:,action]== 0)[0].min()
        self.board[free_slot,action] = 2*int(own)-1
        self.moves_counter += 1

    def is_valid_move(self, action):
        if self.board[-2,action]!=0:
            return False
        return True

    def generate_random_move(self):
        action = np.random.randint(self.width)
        count = 0
        while not self.is_valid_move(action) and count < 10:
            action = np.random.randint(self.width)
            count += 1
        return action

    def step(self, action, *args, **kwargs):
        
        done = False
        info={'outcome':None}
        reward = 0
        
        self.apply_move(int(action))
        self.next_mover = 'opponent'
        
        if self.board[-1,int(action)]==1:
            info={'outcome':'fail'}
            done = True
            reward = -self.rewards[0]
        elif self.check_four_straight(own = True):
            done = True
            info={'outcome':'player'}
            reward = self.rewards[1]
        elif self.moves_counter>= self.max_n_moves:
            done = True
            info={'outcome':'draw'}
            reward = 0
        else:
            opp_move = self.generate_random_move()
            self.apply_move(opp_move, own = False)
            self.next_mover = 'player'
            if self.check_four_straight(own = False):
                done = True
                info={'outcome':'opponent'}
                reward = -self.rewards[1]
            elif self.moves_counter>= self.max_n_moves:
                done = True
                info={'outcome':'draw'}
                reward = 0

        if self.print_out:
            self.render()
            if done:
                print(f' outcome: {info["outcome"]}')            
        return self.get_state(), reward, done, info
    
            
    def get_state(self, own = True, torch_output = False):
        sign = 2*int(own)-1
        if torch_output:
            return torch.tensor(sign*self.board.flatten()).float()
        return sign*self.board.flatten()

    
    def check_four_straight(self, own = True):
        sign = 2*int(own)-1
        four = sign*np.ones((4,), dtype = np.int8)
        
        #self.render()
        for row in self.board:
            if len(search_sequence_numpy(row,four))>0:
                return True
        for column in self.board.T:
            if len(search_sequence_numpy(column,four))>0:
                return True
        r,c =self.board.shape
        diag_idx = range(-(r-1),c)    
        if any([len(search_sequence_numpy(self.board.diagonal(i),sign*four))>0 for i in diag_idx ]):
            return True
        if any([len(search_sequence_numpy(np.fliplr(self.board).diagonal(i),sign*four))>0 for i in diag_idx ]):
            return True
        return False


    def get_four_straight_idxs(self, own = True):
        sign = 2*int(own)-1
        four = sign*np.ones((4,), dtype = np.int8)
        idx = []
        
        for i,row in enumerate(self.board):
            if len(search_sequence_numpy(row,four))>0:
                idx.append([(i,j) for j in search_sequence_numpy(row,four)])
        for j, column in enumerate(self.board.T):
            if len(search_sequence_numpy(column,four))>0:
                idx.append([(i,j) for i in search_sequence_numpy(column,four)])

        r,c =self.board.shape
        diag_idx = range(-(r-3),c-3)
        for i in diag_idx:
            if len(search_sequence_numpy(self.board.diagonal(i),sign*four))>0:
                if i >=0:
                    idx.append([(j,j+i) for j in  search_sequence_numpy(self.board.diagonal(i),sign*four) ])
                else:
                    idx.append([(j-i,j) for j in  search_sequence_numpy(self.board.diagonal(i),sign*four) ])
            if len(search_sequence_numpy(np.fliplr(self.board).diagonal(i),sign*four))>0: 
                if i >= 0:
                    idx.append([(j,self.width-1-j-i) for j in  search_sequence_numpy(np.fliplr(self.board).diagonal(i),sign*four)])
                else:
                    idx.append([(j-i,self.width-1-j) for j in  search_sequence_numpy(np.fliplr(self.board).diagonal(i),sign*four)])
        
        return  [ item for elem in idx for item in elem]

    def get_NN_action(self):
        """  """
        net_output = self.model(self.get_state(own = False, torch_output = True))
        
        action_bool_array = torch.zeros(self.width, dtype=torch.float32)
        action_index = torch.argmax(net_output)
        action_bool_array[action_index] = 1        
        
        return np.where(action_bool_array)[0]


    def get_opponent_action(self):
        """ picks opponent action (random or NN based). If NN one is invalid, returns random"""
        if self.use_NN:
            action = self.get_NN_action()
            if self.is_valid_move(action):
                return action
        if self.use_NN and self.print_out:
            print('NN returned invalid move. Random one is generated!')
        return self.generate_random_move()

        
        

#%%
def sym_rep(symbol):
    """ render symbols """
    if symbol == 0:
        out = ' '
    elif symbol == 1:
        out = 'X'
    elif symbol == -1:
        out = 'O'
    else:
        raise('symbol error')
    return out

def search_sequence_numpy(arr,seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------    
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------    
    Output : 1D Array of indices in the input array that satisfy the 
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []         # No match found

#%%

if __name__ == "__main__":

    game = ConnectFour(print_out=True)
    
    #for i in range(100):
    #    print(f'Game Nr. {i+1}')
    
    game.reset()
    done = False
    
    while not done:
        
        my_action = game.generate_random_move()
        _,_,done,info = game.step(my_action)
        #game.render()
    
    print(info['outcome'])
    if info['outcome']!='fail':
        idxs = game.get_four_straight_idxs(own = (info['outcome']=='player'))
        #print(idxs)
        game.render(idxs)