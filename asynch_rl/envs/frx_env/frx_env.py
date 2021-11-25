#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:53:58 2021

@author: enric
"""

# 4 straight
import numpy as np
import torch
import gym
from gym import spaces
import os
import csv
import random


import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mpdates 


#%%


#leverage 100
#buy/sell options
# lotto = 100k
# dimensione minilotti
#0.01 0.02 0.03 0.05 0.1 0.2 0.5  [ close-buys   hold   close-sells  ]
#commissione 3*10-5 per lotto (all'acquisto')

class FrxTrdr(gym.Env):
    
    def __init__(self, rewards = [100,1], print_out = False, n_samples = 240, \
                 max_n_moves = 1440, initial_account = 10000, fees_type = 'proportional', leverage = 100):
        
        self.env_type = 'Frx'
        self.initial_account = initial_account
        
        
        #self.max_expected_gain = 5
        #self.max_days = 10
        #self.rel_periods = [90, 30,7,3,1]
        
        self.max_open_positions = 20

        self.stop_loss_ratio = 0.75
        self.n_samples = n_samples
        self.leverage = leverage
        
        #coeff_0 = np.array([0.001, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5])
        coeff_0 = np.array([0.001])
        self.coeff = np.append( -np.flipud(coeff_0), np.append(0, coeff_0))
        
        self.n_actions = len(self.coeff)
        self.max_n_moves = max_n_moves
        
        # externally editable params
        self.print_out = print_out
        self.rewards = rewards
        
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1 , self.n_actions-1 ]))  
        
        # 'operations', 'proportional', 'combined'
        self.fees_type = fees_type
        self.fixed_fee = 0.01
        
        self.prop_fee = 3*10**-5
        self.lot_size = 100000
        self.pip = 0.0001


    ###################################
    """ these three methods are needed in sim env / rl env"""
    def get_max_iterations(self):
        return int(self.max_n_moves)
 
    def get_actions_structure(self):
        pass # implemented in wrapper

    def get_observations_structure(self):
        return len(self.get_state())
    ###################################

    def render(self):
        pass    
    
    
    def initialize_database(self):
        
        signal_length = 20000
        segment_length = 5
        average_value = 500
        
        a = 0.3*np.array([0.5,0.3,0.2,0.25])
        
        w = [0.0002,0.0005, 0.001,0.05]
        phi = 2*np.pi*np.random.random(len(a))
        
        raw_signal = average_value+ .02*np.random.randn(signal_length)
        for i in range(len(a)-1):
            raw_signal += a[i]*np.sin(np.arange(signal_length)*w[i]+phi[i])
        
        raw_signal /= average_value
        
        data = None
        date = datetime.now()
        delta = timedelta(minutes = 5)
        
        for i in range(round(signal_length/segment_length)):
            
            section = raw_signal[i*segment_length:(i+1)*segment_length]
            OHLC = np.array([ [ mpdates.date2num(date)  , section[0], np.max(section),np.min(section) , section[-1] ]])
            if data is None:
                data = OHLC
            else:
                data = np.append(data, OHLC, axis = 0)
        
            date = date + delta
    

        #fig,ax = plt.subplots(1,1)
        
        ## plot the candlesticks
        #start = random.randint(0,round(signal_length/segment_length)-50)
        
        #candlestick_ohlc(ax, data[start:start+50,:], width=.002, colorup='green', colordown='red')
        
        self.data = data[:,1:]

    """
    def initialize_database(self, init_row = 0):
        #folders_path = os.path.join(os.getcwd(), 'envs','forex_env' , 'forex_data', 'Train_folder')
        folders_path = os.path.join(os.path.dirname(__file__),'frx_data', 'Train_folder')
        dirs = os.listdir(folders_path)
        selected_dir = random.choice(dirs)
        while not selected_dir in ['EURUSD','USDCHF','USDCAD','EURCHF','EURGBP','EURCAD','USDSEK','USDDKK','USDNOK','USDSGD','USDZAR']:
            selected_dir = random.choice(dirs)
        
        dir_path = os.path.join(folders_path , selected_dir)
        _, _, filenames = next(os.walk(dir_path))
        
        filename = random.choice(filenames)
        #print(f'selected sequence : {filename}')
        
        file_path = os.path.join( dir_path,filename)
        
        file_len = sum(1 for line in open(file_path))
        #file_len =200000
        init_row = np.random.randint(self.n_samples+1, file_len -self.max_n_moves)
        
        self.data = np.loadtxt(file_path, delimiter=',', usecols=(2,3,4,5), unpack=True, dtype=float, skiprows = init_row, max_rows = (self.n_samples+self.max_n_moves+10) ).T
    """

    def reset(self, **kwargs):

        self.positions = {'long': [], 'short': []}
        self.opened_position = False

        self.initialize_database()
        self.current_idx = self.n_samples
        
        self.price = self.data[self.current_idx,3]

        # initialization
        self.account = self.initial_account
        self.session_length = 0
        
        self.balance_hist = []
        
        return self.get_obs()
        
    
    def get_obs(self, decimals_dict = None):
        """ get normalized observed states"""
        # normalize observed samples        
        exch_mean = np.mean(self.data[ self.current_idx-self.n_samples: self.current_idx ,3])
        exch_std = np.std( self.data[ self.current_idx-self.n_samples: self.current_idx , 3 ])
        
        norm_data = (self.data[self.current_idx-self.n_samples: self.current_idx, :] - exch_mean)/exch_std
        #hampel_norm_data = 1/2*(1+np.tanh(0.01*(self.data[self.current_idx-self.n_samples: self.current_idx, :] - exch_mean)/exch_std))
        
        #account_hist_norm = self.account_hist[len(self.account_hist)-self.n_samples:]/(2*self.initial_account)
        #portfolio_hist_norm = self.portfolio_hist[len(self.account_hist)-self.n_samples:]/(2*self.initial_account)
        if decimals_dict is None:
            investments = np.zeros(2,dtype=np.float)
        else:
            investments = np.array( [decimals_dict['long'], np.abs(decimals_dict['short']) ] , dtype=np.float)
        return norm_data , investments


    def get_fee(self, amount):
        if self.fees_type == 'operations':
            return self.fixed_fee*bool(amount)
        elif self.fees_type == 'proportional':
            return self.prop_fee*amount
        else:
            return self.prop_fee*amount + self.fixed_fee*bool(amount)
        

    def open_position(self, decimal):
        """  """
        self.opened_position = True
        amount = np.abs(decimal)*self.lot_size/self.leverage
        if decimal < 0 and len(self.positions['short']) < self.max_open_positions/2:
            self.positions['short'].append([decimal, amount, self.price, amount])
            self.account -= amount + self.prop_fee*np.abs(decimal)*self.lot_size    
        elif decimal >0 and len(self.positions['long']) < self.max_open_positions/2:
            self.positions['long'].append([decimal, amount, self.price, amount])
            self.account -= amount + self.prop_fee*np.abs(decimal)*self.lot_size
        
        

    def close_positions(self, short = False ):
        """ """
        if short:
            typ = 'short'
        else:
            typ = 'long'
        for decimal, amount, price, investment_value in self.positions[typ]:
            self.account += investment_value
        self.positions[typ] = []


    def update_open_positions_value(self):
        """ """
        total_investment = 0
        total_decimals = {'short':0, 'long':0}
        for typ in ['short', 'long']:
            for i, (decimal, amount, price, _) in enumerate(self.positions[typ]):
                investment = amount + decimal*self.lot_size*(self.price - price)/price
                self.positions[typ][i][3] = investment
                total_investment += investment
                total_decimals[typ] += decimal
        return total_investment, total_decimals


    def plot_graphs(self):
        
        fig, ax = plt.subplots(1,1)
        ax.plot(self.balance_hist)
        

    def step(self, action, *args, **kwargs):
        # here we have to implement the effect of the action on the portfolio
        
        done = False
        info={}
        reward = 0
        action = action.astype(np.int8)
        
        # action[0] -> -1 (close shorts), 0 (do nothing), 1 (close longs)
        # self.coeff[action[1]] -> -x (open short position), 0 (do nothing), x (open long position position)
        
        if np.abs(action[0]) > 0:
            self.close_positions(short = action[0]<0)

        if np.abs(self.coeff[action[1]]) >= 0.01:
            self.open_position(self.coeff[action[1]])

        self.current_idx +=1
        self.session_length += 1

        previous_price = self.price
        self.price = self.data[self.current_idx,3]

        total_investment, decimals_dict = self.update_open_positions_value() 
        self.balance_hist.append(self.account + total_investment)

        if self.session_length > self.max_n_moves :
            self.close_positions()
            self.close_positions(short = True)
        
        if  self.opened_position and all([v==[] for k,v in self.positions.items()]):
            done = True
            final_return = 100*(self.account - self.initial_account ) / self.initial_account
            reward = final_return
            #reward =   np.sign(final_return)*(100*final_return)**2
            info['steps'] = self.session_length
            info['return'] = final_return
       
        return self.get_obs(decimals_dict) , reward[0] if isinstance(reward,np.ndarray) else reward, done, info
    


#%%

import matplotlib.pyplot as plt

if __name__ == "__main__":

    env = FrxTrdr()
    
    state = env.reset()
    #print(state.shape)
    
    done = False

    
    while not done:
        
        action = np.zeros(env.n_actions, dtype = np.bool)
        action[np.random.randint(env.n_actions)] = True
        state, reward ,done,info = env.step(action)
        #print(state.shape)

    print(f'final balance: {np.round(reward, 4)}')
    print(f'total steps  : {info["steps"]}')
    
    fig, ax = plt.subplots(1,1)
    
    ax.plot(env.balance_hist)
    
