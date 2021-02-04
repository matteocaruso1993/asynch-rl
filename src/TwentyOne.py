#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:33:24 2021

@author: Enrico Regolin
"""

#%%
from nns.custom_networks import LinearModel
from rl.resources.memory import ReplayMemory, unpack_batch

import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm



#%%

class TwentyOne():
    def __init__(self, reward = [-2, 5, 7, 8, 20] , threshold = 21):
        self.state = np.zeros(5)
        self.reward = reward
        self.threshold = threshold
        
    def reset(self):
        self.state = np.sort(np.random.randint(1,7,5))
        self.n_rethrows = 0
        
    def step(self, action):
        
        info = {}
        done = False
        
        if any(action):
            rethrow = np.random.randint(1,7,np.sum(action))
            self.state[action] = rethrow
            self.state = np.sort(self.state)
            reward = -np.sum(action)
            self.n_rethrows += 1
        else:
            reward = 0
            done = True
            info['result'] = 0
        
        success = self.success_condition() 
        
        #if not success and self.n_rethrows == 1:
        #    stophere=1
        
        if success or self.n_rethrows >= 2:
            done = True
            reward += self.reward[success]
            info['result'] = success
            #if success:
            #    print(self.state)
        
        return self.state/6, reward, done, info


    def success_condition(self):
        if np.sum(self.state) >= self.threshold:
            if np.max(np.bincount(self.state)[1:]) >= 3:
                return 1
        if np.equal(np.sort(self.state) , np.array([1,2,3,4,5]) ).all() or np.equal(np.sort(self.state) , np.array([2,3,4,5,6]) ).all():
            return 2
        if any([np.sum(self.state == i)==4 for i in range(1,7)]) :
            return 3
        if any([np.all(self.state == i) for i in range(1,7)]) :
            return 4
        return 0
    
    
def qv_update(mem, model_qv):
    
    state_batch, action_batch, reward_batch, state_1_batch, done_batch, _ = mem.extractMinibatch()

    with torch.no_grad():
        output_1_batch = model_qv(state_1_batch.float())

    y_batch = torch.cat(tuple(reward_batch[i].unsqueeze(0) if done_batch[i]  #minibatch[i][4]
                              else (reward_batch[i] + 0.99 * torch.max(output_1_batch[i])).unsqueeze(0)
                              for i in range(len(reward_batch))))
    q_value = torch.sum(model_qv(state_batch.float()) * action_batch, dim=1)
    
    model_qv.optimizer.zero_grad()
    y_batch = y_batch.detach()
    loss_qval = model_qv.criterion_MSE(q_value, y_batch)
    loss_qval.backward()
    model_qv.optimizer.step()  
    
    loss_qv = loss_qval.item()
    
    return loss_qv
    

def extract_tensor_batch(t, batch_size):
    # extracs batch from first dimension (only works for 2D tensors)
    idx = torch.randperm(t.nelement())
    return t.view(-1)[idx][:batch_size].view(batch_size,1)


#%%
"""
if __name__ == "__main__":
    
    # hyperparams
    tot_iter = 10000    
    n_iter = 500
    gamma = 0.99
    beta = .01

    # init
    loss_qv_i = 0
    loss_pg_i = 0
    
    n_actions = 32
    prob_even = 1/n_actions*torch.ones(n_actions)
    entropy_max = np.round(torch.sum(-torch.log(prob_even)*prob_even).item(),3)
    
    actor =  LinearModel(0, 'LinearModel0', 0.0001, n_actions, 5, *[10,50], softmax=True ) 
    critic = LinearModel(1, 'LinearModel1', 0.0001, 1, 5, *[10,50] ) 
    
    game = TwentyOne()
    
    reward_history =[]
    loss_hist = []
    loss_qv_hist = []
    
    entropy_history = []
    retries_history = []
    action_history = []

    results_array = np.zeros((1,5))
    results = []
    
    for iteration in tqdm(range(tot_iter)):
        
        done = False
        game.reset()
        state = game.state/6
        
        cum_reward = 0
        loss_policy = 0
        advantage_loss = 0
        tot_rethrows = 0

        actor.optimizer.zero_grad()
        critic.optimizer.zero_grad()
        
        traj_entropy = []
        traj_rewards = []
        traj_state_val = []
        traj_log_prob = []
        state_sequences = []
        
        while not done:
            
            state_in = torch.tensor(state).unsqueeze(0)
            state_sequences.append(state_in)
            
            with torch.no_grad():
                state_val = critic(state_in.float())
            
            prob = actor(state_in.float())
                
            entropy = torch.sum(-torch.log(prob)*prob)
            
            action_idx = torch.multinomial(prob , 1, replacement=True).item()
            action_string = '{0:05b}'.format(action_idx)
            action = np.flipud(np.array([bool(int(c)) for c in action_string]))
            
            tot_rethrows += np.sum(action)
            
            state_1, reward, done, info = game.step(action)
            
            cum_reward += reward
            
            traj_log_prob.append(prob[:,action_idx])
            traj_state_val.append(state_val)
            traj_rewards.append(reward)
            traj_entropy.append(entropy)
                    
            state = state_1
 
        R = 0
        for i in range(len(traj_rewards)):
            R = traj_rewards[-1-i] + gamma* R
            advantage = R - traj_state_val[-1-i]
            loss_policy += -advantage*traj_log_prob[-1-i] + beta*traj_entropy[-1-i]
            advantage_loss += (  R - critic( state_sequences[-1-i].float()) )**2          
            
        total_loss = loss_policy + advantage_loss
     
        total_loss.backward()
        actor.optimizer.step()
        critic.optimizer.step()
        
        loss_pg_i = np.round(loss_policy.item(),3)
        loss_qv_i = np.round(advantage_loss.item(),3)

        loss_hist.append(loss_pg_i)
        loss_qv_hist.append(loss_qv_i)
        
        action_history.append(tot_rethrows)
        retries_history.append(game.n_rethrows)

        results.append(info['result'])        
        reward_history.append(cum_reward)
        entropy_history.append( round(torch.mean(torch.cat([t.unsqueeze(0) for t in traj_entropy])).item(),2) )
        
        if iteration> 0 and not iteration%n_iter:
            print('last {n_iter} averages')
            
            print(f'loss Actor {round( sum(abs(np.array(loss_hist[-n_iter:])))/n_iter )}')
            print(f'loss Critic {round( sum(loss_qv_hist[-n_iter:])/n_iter )}')
            print(f'entropy {round( 100*sum(entropy_history[-n_iter:])/entropy_max/n_iter,2)}%')
            print(f'cum reward {round( sum(reward_history[-n_iter:])/n_iter , 2)}')
            #print(f'duration {round(sum( duration_hist[-n_iter:]) / n_iter )}')       
        
    results_array = results_array[1:,:]

    fig0, ax0 = plt.subplots(3,1)
    
    ax0[0].plot(loss_qv_hist)
    ax0[1].plot(loss_hist)
    ax0[2].plot(entropy_history)


    fig, ax = plt.subplots(4,1)
    N = 100

    #ax[1].plot(reward_history)
    ax[0].plot(np.zeros(len(reward_history)))
    ax[0].plot(np.convolve(reward_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])
    ax[0].plot(0.5*np.ones(len(reward_history)))
    
    ax[1].plot(action_history)
    ax[1].plot(np.convolve(action_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])

    ax[2].plot(retries_history)
    ax[2].plot(np.convolve(retries_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])
    
    
    ax[3].stackplot(np.arange(results_array.shape[0]) , results_array.T, labels=['lost','Tris+21','Straight','PokerDice','WWWWW'])
    plt.legend(loc='upper left')

"""
#%%
# single Net setting

if __name__ == "__main__":
    
    # hyperparams
    tot_iter = 10000    
    n_iter = 500
    gamma = 0.99
    beta = .01

    # init
    loss_qv_i = 0
    loss_pg_i = 0
    
    n_actions = 32
    prob_even = 1/n_actions*torch.ones(n_actions)
    entropy_max = np.round(torch.sum(-torch.log(prob_even)*prob_even).item(),3)
    
    #actor =  LinearModel(0, 'LinearModel0', 0.0001, n_actions, 5, *[10,10], softmax=True ) 
    #critic = LinearModel(1, 'LinearModel1', 0.0001, 1, 5, *[10,10] ) 
    model = LinearModel(1, 'LinearModel1', 0.0001, n_actions+1, 5, *[10,100] )
    sm = nn.Softmax(dim = 1)
    
    game = TwentyOne()
    
    reward_history =[]
    loss_hist = []
    loss_qv_hist = []
    
    entropy_history = []
    retries_history = []
    action_history = []

    results_array = np.zeros((1,5))
    results = []
    
    for iteration in tqdm(range(tot_iter)):
        
        done = False
        game.reset()
        state = game.state/6
        
        cum_reward = 0
        loss_policy = 0
        advantage_loss = 0
        tot_rethrows = 0

        model.optimizer.zero_grad()
        #actor.optimizer.zero_grad()
        #critic.optimizer.zero_grad()
        
        traj_entropy = []
        traj_rewards = []
        traj_state_val = []
        traj_log_prob = []
        state_sequences = []
        
        while not done:
            
            state_in = torch.tensor(state).unsqueeze(0)
            state_sequences.append(state_in)
            
            with torch.no_grad():
                state_val = model(state_in.float())[:,-1]
            
            prob = sm(model(state_in.float())[:,:-1])
                
            entropy = torch.sum(-torch.log(prob)*prob)
            
            action_idx = torch.multinomial(prob , 1, replacement=True).item()
            action_string = '{0:05b}'.format(action_idx)
            action = np.flipud(np.array([bool(int(c)) for c in action_string]))
            
            tot_rethrows += np.sum(action)
            
            state_1, reward, done, info = game.step(action)
            
            cum_reward += reward
            
            traj_log_prob.append(prob[:,action_idx])
            traj_state_val.append(state_val)
            traj_rewards.append(reward)
            traj_entropy.append(entropy)
                    
            state = state_1
 
        R = 0
        for i in range(len(traj_rewards)):
            R = traj_rewards[-1-i] + gamma* R
            advantage = R - traj_state_val[-1-i]
            loss_policy += -advantage*traj_log_prob[-1-i] + beta*traj_entropy[-1-i]
            advantage_loss += (  R - model( state_sequences[-1-i].float())[:,-1] )**2          
            
        total_loss = loss_policy + advantage_loss
     
        total_loss.backward()
        model.optimizer.step()
        #actor.optimizer.step()
        #critic.optimizer.step()
        
        loss_pg_i = np.round(loss_policy.item(),3)
        loss_qv_i = np.round(advantage_loss.item(),3)

        loss_hist.append(loss_pg_i)
        loss_qv_hist.append(loss_qv_i)
        
        action_history.append(tot_rethrows)
        retries_history.append(game.n_rethrows)

        results.append(info['result'])        
        reward_history.append(cum_reward)
        entropy_history.append( round(torch.mean(torch.cat([t.unsqueeze(0) for t in traj_entropy])).item(),2) )
        
        if iteration> 0 and not iteration%n_iter:
            print('last {n_iter} averages')
            
            print(f'loss Actor {round( sum(abs(np.array(loss_hist[-n_iter:])))/n_iter )}')
            print(f'loss Critic {round( sum(loss_qv_hist[-n_iter:])/n_iter )}')
            print(f'entropy {round( 100*sum(entropy_history[-n_iter:])/entropy_max/n_iter,2)}%')
            print(f'cum reward {round( sum(reward_history[-n_iter:])/n_iter , 2)}')
            #print(f'duration {round(sum( duration_hist[-n_iter:]) / n_iter )}')       
        
    results_array = results_array[1:,:]

    fig0, ax0 = plt.subplots(3,1)
    
    ax0[0].plot(loss_qv_hist)
    ax0[1].plot(loss_hist)
    ax0[2].plot(entropy_history)


    fig, ax = plt.subplots(4,1)
    N = 100

    #ax[1].plot(reward_history)
    ax[0].plot(np.zeros(len(reward_history)))
    ax[0].plot(np.convolve(reward_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])
    ax[0].plot(0.5*np.ones(len(reward_history)))
    
    ax[1].plot(action_history)
    ax[1].plot(np.convolve(action_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])

    ax[2].plot(retries_history)
    ax[2].plot(np.convolve(retries_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])
    
    
    ax[3].stackplot(np.arange(results_array.shape[0]) , results_array.T, labels=['lost','Tris+21','Straight','PokerDice','WWWWW'])
    plt.legend(loc='upper left')

    
