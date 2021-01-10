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
    def __init__(self, reward = [9,-1] , threshold = 21):
        self.state = np.zeros(5)
        self.reward = reward
        self.threshold = threshold
        
    def reset(self):
        self.state = np.sort(np.random.randint(1,7,5))
        self.n_rethrows = 0
        
    def step(self, action):
        
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
        
        success = self.success_condition() 
        
        #if not success and self.n_rethrows == 1:
        #    stophere=1
        
        if success or self.n_rethrows >= 2:
            done = True
            reward += self.reward[int(not success)]
            #if success:
            #    print(self.state)
        
        return self.state/6, reward, done, {}


    def success_condition(self):
        if np.sum(self.state) >= self.threshold:
            if np.max(np.bincount(self.state)[1:]) >= 3:
                return True
        return False
    
    
def qv_update(mem, model_qv):
    
    state_batch, action_batch, reward_batch, state_1_batch, done_batch, _ = mem.extractMinibatch()

    if move_to_cuda:  # put on GPU if CUDA is available
        state_batch = state_batch.cuda()
        action_batch = action_batch.cuda()
        reward_batch = reward_batch.cuda()
        state_1_batch = state_1_batch.cuda()

    # get output for the next state
    with torch.no_grad():
        output_1_batch = model_qv(state_1_batch.float())
    # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
    y_batch = torch.cat(tuple(reward_batch[i].unsqueeze(0) if done_batch[i]  #minibatch[i][4]
                              else (reward_batch[i] + 0.99 * torch.max(output_1_batch[i])).unsqueeze(0)
                              for i in range(len(reward_batch))))
    # extract Q-value
    # calculates Q value corresponding to all actions, then selects those corresponding to the actions taken
    q_value = torch.sum(model_qv(state_batch.float()) * action_batch, dim=1)
    
    model_qv.optimizer.zero_grad()
    y_batch = y_batch.detach()
    loss_qval = model_qv.criterion_MSE(q_value, y_batch)
    loss_qval.backward()
    model_qv.optimizer.step()  
    
    loss_qv = loss_qval.item()
    
    return loss_qv
    

def offline_pg_update(memory, model_pg, model_qv, beta, n_epochs):
    
    # extract all
    state_batch, action_batch, reward_batch, state_1_batch, done, actual_idxs_batch = unpack_batch( memory )

    if move_to_cuda:  # put on GPU if CUDA is available
        state_batch = state_batch.cuda()
        action_batch = action_batch.cuda()
        reward_batch = reward_batch.cuda()
        state_1_batch = state_1_batch.cuda()

    # we re-compute the probabilities, this time in batch (they have to match with the outputs obtained during the simulation)
    prob_distribs_batch = model_pg(state_batch.float())   
    #action_idx_batch = torch.argmax(prob_distribs_batch, dim = 1)
    action_idx_batch = actual_idxs_batch
    
    action_batch_grad = torch.zeros(action_batch.shape).cuda()
    action_batch_grad[torch.arange(action_batch_grad.size(0)),action_idx_batch] = 1
        
    prob_action_batch = prob_distribs_batch[torch.arange(prob_distribs_batch.size(0)), action_idx_batch].unsqueeze(1)
    entropy = -torch.sum(prob_distribs_batch*torch.log(prob_distribs_batch),dim = 1).unsqueeze(1)
    
    with torch.no_grad():
        val_1 = model_qv(state_1_batch.float())
        val_0 = model_qv(state_batch.float())
        
    #advantage = reward_batch + 0.99 *torch.max(val_1, dim = 1, keepdim = True)[0]\
    #        - torch.max(val_0, dim = 1, keepdim = True)[0]

    advantage = (reward_batch + 0.99 *torch.max(val_1, dim = 1)[0]\
            - torch.max(val_0, dim = 1)[0]).unsqueeze(1)

    loss_vector =  -torch.log(prob_action_batch)*advantage + beta*(entropy-1)**2


    if n_epochs < 10:
        model_pg.optimizer.zero_grad()
        loss_policy = torch.sum(loss_vector)
        loss_policy.backward()
        model_pg.optimizer.step()  
    else:
        losses = []
        batch_size = min(mem.minibatch_size, round(0.5*loss_vector.shape[0]))
        model_pg.optimizer.zero_grad()
        
        for _ in range(n_epochs):
            #self.model_pg.optimizer.zero_grad()
            loss = torch.mean(extract_tensor_batch(loss_vector, batch_size))
            loss.backward(retain_graph = True)
            losses.append(loss.unsqueeze(0))
            #self.model_pg.optimizer.step() 
        # last run erases the gradient
        #self.model_pg.optimizer.zero_grad()
        loss = torch.mean(extract_tensor_batch(loss_vector, batch_size))
        loss.backward(retain_graph = False)
        losses.append(loss.unsqueeze(0))
        model_pg.optimizer.step()  
        
        loss_policy = torch.mean(torch.cat(losses))

    return loss_policy.item()


def offline_update(memory, model, beta, n_epochs):
    
    sm = nn.Softmax(dim = 1)   
    
    # extract all
    state_batch, action_batch, reward_batch, state_1_batch, done_batch, actual_idxs_batch = unpack_batch( memory )

    if move_to_cuda:  # put on GPU if CUDA is available
        state_batch = state_batch.cuda()
        action_batch = action_batch.cuda()
        reward_batch = reward_batch.cuda()
        state_1_batch = state_1_batch.cuda()

    # we re-compute the probabilities, this time in batch (they have to match with the outputs obtained during the simulation)
    val_0 = model(state_batch.float())
    val_1 = model(state_1_batch.float())
    
    prob_distribs_batch = sm(val_0)

    #action_idx_batch = torch.argmax(prob_distribs_batch, dim = 1)
    action_idx_batch = actual_idxs_batch
    
    action_batch_grad = torch.zeros(action_batch.shape).cuda()
    action_batch_grad[torch.arange(action_batch_grad.size(0)),action_idx_batch] = 1
        
    prob_action_batch = prob_distribs_batch[torch.arange(prob_distribs_batch.size(0)), action_idx_batch].unsqueeze(1)
    entropy = -torch.sum(prob_distribs_batch*torch.log(prob_distribs_batch),dim = 1).unsqueeze(1)
    

    advantage = (reward_batch + 0.99 *torch.max(val_1, dim = 1)[0]\
            - torch.max(val_0, dim = 1)[0]).unsqueeze(1)

    pg_loss_vector =  -torch.log(prob_action_batch)*advantage.detach() + beta*(entropy-1)**2


    # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
    y_batch = torch.cat(tuple(reward_batch[i].unsqueeze(0) if done_batch[i]  #minibatch[i][4]
                              else (reward_batch[i] + 0.99 * torch.max(val_1[i])).unsqueeze(0)
                              for i in range(len(reward_batch))))
    # extract Q-value
    # calculates Q value corresponding to all actions, then selects those corresponding to the actions taken
    q_value = torch.sum(val_0 * action_batch, dim=1)
    loss_qval = (q_value - y_batch.detach())**2
    
    total_loss_vector = loss_qval + pg_loss_vector

    losses = []
    batch_size = min(mem.minibatch_size, round(0.5*total_loss_vector.shape[0]))
    model.optimizer.zero_grad()
    
    for _ in range(n_epochs):
        model.optimizer.zero_grad()
        loss = torch.mean(extract_tensor_batch(total_loss_vector, batch_size))
        loss.backward(retain_graph = False)  # True!
        losses.append(loss.unsqueeze(0))
        model.optimizer.step() 

    # last run erases the gradient
    model.optimizer.zero_grad()
    loss = torch.mean(extract_tensor_batch(total_loss_vector, batch_size))
    loss.backward(retain_graph = False)
    losses.append(loss.unsqueeze(0))
    model.optimizer.step()  
    
    loss_policy = torch.mean(torch.cat(losses))

    return torch.mean(pg_loss_vector).item(), torch.mean(loss_qval).item()


def extract_tensor_batch(t, batch_size):
    # extracs batch from first dimension (only works for 2D tensors)
    idx = torch.randperm(t.nelement())
    return t.view(-1)[idx][:batch_size].view(batch_size,1)

#%%
"""
if __name__ == "__main__":

    sm = nn.Softmax(dim = 1)    

    loss_qv_i = 0
    loss_pg_i = 0

    n_epochs = 100
    update_freq = 200

    beta = 0.1
    
    move_to_cuda = True
    model = LinearModel(0, 'LinearModel0', 0.0001, 32, 5, *[10, 10] ) 
    if move_to_cuda:
        model = model.cuda()
    
    
    mem = ReplayMemory(size = 600, minibatch_size= 64)
    
    game = TwentyOne()
    
    reward_history =[]
    loss_hist = []
    loss_qv_hist = []
    
    entropy_history = []
    retries_history = []
    action_history = []

    
    for iteration in tqdm(range(25000)):
        
        
        done = False
        game.reset()
        state = game.state/6
        
        cum_reward = 0
        loss_policy = 0
        tot_rethrows = 0
        
        traj_entropy = []
        
        while not done:
            
            state_in = torch.tensor(state).unsqueeze(0)
            
            q_val = model(state_in.float().cuda()).cpu()
            prob = sm(q_val)
            
            entropy = torch.sum(-torch.log(prob)*prob)
            
            action_idx = torch.multinomial(prob , 1, replacement=True).item()
            action_string = '{0:05b}'.format(action_idx)
            action = np.flipud(np.array([bool(int(c)) for c in action_string]))
            
            tot_rethrows += np.sum(action)
            
            state_1, reward, done, _ = game.step(action)
            cum_reward += reward
            
            action_qv = torch.zeros(1,32)
            action_qv[:,action_idx] = 1
            
            mem.addInstance((state_in, action_qv, torch.tensor(reward).float().unsqueeze(0), \
                                 torch.tensor(state_1).unsqueeze(0), done, action_idx))
                    
            state = state_1
            traj_entropy.append(entropy.item())
            
        if not iteration % update_freq and iteration > 0:
            print('')
            print(f'average reward, last {update_freq}: {round(sum(reward_history[-update_freq:]) / update_freq,2)}')
            print('')
            
            loss_pg_i, loss_qv_i = offline_update(mem.getMemoryAndEmpty(), model, beta, n_epochs) 

        loss_hist.append(loss_pg_i)
        loss_qv_hist.append(loss_qv_i)
        
        
        action_history.append(tot_rethrows)
        retries_history.append(game.n_rethrows)
        
        reward_history.append(cum_reward)
        entropy_history.append( round(sum(traj_entropy) / len(traj_entropy),2) )
            
        
        

    fig0, ax0 = plt.subplots(3,1)
    
    ax0[0].plot(loss_qv_hist)
    ax0[1].plot(loss_hist)
    ax0[2].plot(entropy_history)


    fig, ax = plt.subplots(3,1)
    N = 100

    #ax[1].plot(reward_history)
    ax[0].plot(np.zeros(len(reward_history)))
    ax[0].plot(np.convolve(reward_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])
    ax[0].plot(0.5*np.ones(len(reward_history)))
    
    ax[1].plot(action_history)
    ax[1].plot(np.convolve(action_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])

    ax[2].plot(retries_history)
    ax[2].plot(np.convolve(retries_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])
    
"""

#%%

if __name__ == "__main__":

    tot_iter = 50000    

    offline_pg = True
    actor_critic = True
    qv_update_started = False

    loss_qv_i = 0
    loss_pg_i = 0
    
    n_epochs_pg = 100
    n_epochs = 200
    qv_freq = 1000
    pg_freq = 200

    beta = 0.1
    
    move_to_cuda = True
    model_pg = LinearModel(0, 'LinearModel0', 0.0001, 32, 5, *[10, 10] , softmax = True) 
    if move_to_cuda and offline_pg:
        model_pg = model_pg.cuda()

    model_qv = LinearModel(0, 'LinearModel1', 0.0001, 32, 5, *[10,10] ) 
    if move_to_cuda:
        model_qv = model_qv.cuda()
    
    mem = ReplayMemory(size = 3000, minibatch_size= 512)
    
    mem_pg = ReplayMemory(size = 600, minibatch_size= 64)
    
    game = TwentyOne()
    
    reward_history =[]
    loss_hist = []
    loss_qv_hist = []
    
    entropy_history = []
    retries_history = []
    action_history = []

    
    for iteration in tqdm(range(tot_iter)):
        
        
        done = False
        game.reset()
        state = game.state/6
        
        cum_reward = 0
        loss_policy = 0
        tot_rethrows = 0
        
        
        traj_entropy = []
        
        if not offline_pg:
            model_pg.optimizer.zero_grad()
        
        while not done:
            
            state_in = torch.tensor(state).unsqueeze(0)
            
            if not (offline_pg and move_to_cuda):
                prob = model_pg(state_in.float())
            else:
                prob = model_pg(state_in.float().cuda()).cpu()
                
            entropy = torch.sum(-torch.log(prob)*prob)
            
            action_idx = torch.multinomial(prob , 1, replacement=True).item()
            action_string = '{0:05b}'.format(action_idx)
            action = np.flipud(np.array([bool(int(c)) for c in action_string]))
            
            tot_rethrows += np.sum(action)
            
            state_1, reward, done, _ = game.step(action)
            cum_reward += reward
            
            action_qv = torch.zeros(1,32)
            action_qv[:,action_idx] = 1
            
            if actor_critic:
                mem.addInstance((state_in, action_qv, torch.tensor(reward).float().unsqueeze(0), \
                                 torch.tensor(state_1).unsqueeze(0), done, action_idx))
                    
            if offline_pg:
                mem_pg.addInstance((state_in, action_qv, torch.tensor(reward).float().unsqueeze(0), \
                                 torch.tensor(state_1).unsqueeze(0), done, action_idx))
                    
            state = state_1
            traj_entropy.append(entropy.item())
            
            if not offline_pg:
                if not actor_critic or not qv_update_started:
                    loss_policy += -cum_reward*torch.log(prob[:,action_idx]) 
                else:
                    with torch.no_grad():
                        advantage = reward + 0.99 * torch.max(model_qv(torch.tensor(state_1).unsqueeze(0).float().cuda())) \
                            - torch.max(model_qv(state_in.float().cuda()))
                    loss_policy += -advantage.cpu()*torch.log(prob[:,action_idx]) 
                
                loss_policy += beta*(entropy-1)**2
           
        if not offline_pg:
            loss_policy.backward()
            model_pg.optimizer.step()
            loss_pg_i = np.round(loss_policy.item(),3)
            
        if offline_pg and not iteration % pg_freq and iteration > 0:
            loss_pg_i = offline_pg_update(mem_pg.getMemoryAndEmpty(), model_pg, model_qv, beta, n_epochs_pg) 

        loss_hist.append(loss_pg_i)
        
        
        action_history.append(tot_rethrows)
        retries_history.append(game.n_rethrows)
        
        reward_history.append(cum_reward)
        entropy_history.append( round(sum(traj_entropy) / len(traj_entropy),2) )
        
            
        
        if actor_critic and mem.isFull() and not iteration % qv_freq:
            print('')
            print(f'average reward, last {qv_freq}: {round(sum(reward_history[-qv_freq:]) / qv_freq,2)}')
            print('')
            
            qv_update_started = True
            loss_qv_batch = []
            for i in range(n_epochs):
            
                loss_qv = qv_update(mem, model_qv)
                loss_qv_batch.append(loss_qv)
            
            loss_qv_i = sum(loss_qv_batch) / n_epochs
            
        loss_qv_hist.append(loss_qv_i)
        
        

    fig0, ax0 = plt.subplots(3,1)
    
    ax0[0].plot(loss_qv_hist)
    ax0[1].plot(loss_hist)
    ax0[2].plot(entropy_history)


    fig, ax = plt.subplots(3,1)
    N = 100

    #ax[1].plot(reward_history)
    ax[0].plot(np.zeros(len(reward_history)))
    ax[0].plot(np.convolve(reward_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])
    ax[0].plot(0.5*np.ones(len(reward_history)))
    
    ax[1].plot(action_history)
    ax[1].plot(np.convolve(action_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])

    ax[2].plot(retries_history)
    ax[2].plot(np.convolve(retries_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])
    


    
#%%

"""

game.reset()
print(game.state)
print(f'sum = {np.sum(game.state)}')


state_in = torch.tensor(game.state/6).unsqueeze(0)

prob = model_pg(state_in.float())
entropy = torch.sum(-torch.log(prob)*prob)
action = torch.multinomial(prob, 1, replacement=True).item()
print(np.round(prob.detach().numpy(),3))

val = model_qv(state_in.float().cuda())

print(val)
"""
#%%
#state_1, reward, done, _ = game.step(action)



                
                