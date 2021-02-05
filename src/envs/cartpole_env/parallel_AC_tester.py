#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:55:47 2021

@author: rodpod21
"""

# parallel AC tester

import numpy as np
import time
from  CartPole_env import CartPoleEnv

from copy import deepcopy
# for AC training
from nns.custom_networks import LinearModel
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


#####
from argparse import ArgumentParser

parser = ArgumentParser()

#following params always to be declared
parser.add_argument("-n", "--n-agents", dest="n_agents", type=int, default=1,
                    help="number of agents")

parser.add_argument("-s", "--save-data", dest="save_data", type=bool, default=True,
                    help="Save sim results")

parser.add_argument("-tot", "--tot-iterations", dest="tot_iterations", type=int, default=25000,
                    help="totl number of iterations")

parser.add_argument("-p", "--generate-plots", dest="generate_plots", type=bool, default=False,
                    help="generate plots")

args = parser.parse_args()

#####


def main(n_agents = 4, generate_plots = False, save_data = True, tot_iterations = 25000):

    
    #n_agents_vector = [1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,100]
    
    t0 = time.time()
    #continue_training = False
    
    n_agents = 4
    
    # hyperparams
    n_epochs = int(np.round(tot_iterations/n_agents))    
    
    gamma = 0.95
    beta = 0.1
    lr = 0.001
    
    sm = nn.Softmax(dim = 1)
    
    loss_qv_i = 0
    loss_pg_i = 0
    
    display_iter = 200
    
    n_actions = 17
    
    prob_even = 1/n_actions*torch.ones(n_actions)
    entropy_max = np.round(torch.sum(-torch.log(prob_even)*prob_even).item(),3)
    game = CartPoleEnv()
    action_pool = np.linspace(-1,1,n_actions)*game.max_force
    
    #if not continue_training:
    
    # initialize model
    ACTOR =  LinearModel(-1, 'LinearModel0', lr, n_actions , 5, *[50,50], softmax=True ) 
    CRITIC = LinearModel(-1, 'LinearModel1', lr, 1, 5, *[50,50] ) 
    #ACTOR.optimizer = torch.optim.SGD(ACTOR.parameters(), lr=lr)
    #CRITIC.optimizer = torch.optim.SGD(CRITIC.parameters(), lr=lr)
    #MODEL = LinearModel(0, 'LinearModel0', 0.0005, n_actions+1 , 5, *[100,100])
    ACTOR.init_weights()
    CRITIC.init_weights()
    
    state = game.reset()
    state_in = torch.tensor(state).unsqueeze(0)
    a = ACTOR(state_in.float())
    b = CRITIC(state_in.float())
    fake_loss = torch.sum(a)**2 +torch.sum(b)**2
    fake_loss.backward()
    ACTOR.optimizer.zero_grad()
    CRITIC.optimizer.zero_grad()

    reward_history =[]
    loss_hist = []
    loss_qv_hist = []
    duration_hist = []
    entropy_history = []
    
    ACTOR_initial_weights = deepcopy(ACTOR.state_dict())
    CRITIC_initial_weights = deepcopy(CRITIC.state_dict())

    reward_iter =[]
    loss_pg_iter = []
    loss_qv_iter = []
    duration_iter = []
    entropy_iter = []
    
    
    for epoch in range(n_epochs):
    
        #clone models
        actors = []
        critics = []
        for i in range(n_agents):
            act = LinearModel(i, 'LinearModel0', lr, n_actions , 5, *[50,50], softmax=True ) 
            crt = LinearModel(i, 'LinearModel0', lr, 1 , 5, *[50,50]) 
            act.load_state_dict(ACTOR.state_dict())
            crt.load_state_dict(CRITIC.state_dict())
            actors.append(deepcopy(act))
            critics.append(deepcopy(crt))
        
    
        for n in range(n_agents):
           
            done = False
            
            state = game.reset()
            
            cum_reward = 0
            loss_policy = 0
            advantage_loss = 0
            tot_rethrows = 0
    
            actors[n].optimizer.zero_grad()
            critics[n].optimizer.zero_grad()
            
            traj_entropy = []
            traj_rewards = []
            traj_state_val = []
            traj_log_prob = []
            state_sequences = []
            
            while not done:
                
                state_in = torch.tensor(state).unsqueeze(0)
                state_sequences.append(state_in)
                
                with torch.no_grad():
                    state_val = critics[n](state_in.float())
                
                prob = actors[n](state_in.float())
                    
                entropy = torch.sum(-torch.log(prob)*prob)
                
                try:
                    action_idx = torch.multinomial(prob , 1, replacement=True).item()
                except Exception:
                    action_idx = torch.argmax(prob).item()
                    
                action = np.array([action_pool[action_idx]])
                
                state_1, reward, done, info = game.step(action)
                
                cum_reward += reward
                
                traj_log_prob.append(torch.log(prob)[:,action_idx])
                traj_state_val.append(state_val)
                traj_rewards.append(reward)
                traj_entropy.append(entropy)
                      
                state = state_1
     
            R = 0
            for i in range(len(traj_rewards)):
                R = traj_rewards[-1-i] + gamma* R
                advantage = R - traj_state_val[-1-i]
                loss_policy += -advantage*traj_log_prob[-1-i] - beta*traj_entropy[-1-i]
                advantage_loss += (  R - critics[n]( state_sequences[-1-i].float()) )**2       
                total_loss = loss_policy + advantage_loss
    
            total_loss.backward() # here we have accumulated the gradients in the model
    
    
    
            duration_iter.append(len(traj_rewards))       
     
            loss_pg_i = np.round(loss_policy.item(),3)
            loss_qv_i = np.round(advantage_loss.item(),3)
    
            loss_pg_iter.append(loss_pg_i)
            loss_qv_iter.append(loss_qv_i)
    
            reward_iter.append(cum_reward)
            entropy_iter.append( round(torch.mean(torch.cat([t.unsqueeze(0) for t in traj_entropy])).item(),2) )
    
        
        for actor in actors:
            for net1,net2 in zip(actor.named_parameters(),ACTOR.named_parameters()):
                net2[1].grad += net1[1].grad.clone()/n_agents
    
        for critic in critics:
            for net1,net2 in zip(critic.named_parameters(),CRITIC.named_parameters()):
                net2[1].grad += net1[1].grad.clone()/n_agents
                            
         
        # here we have transferred the gradients to the main model
        ACTOR.optimizer.step()
        ACTOR.optimizer.zero_grad()
    
        CRITIC.optimizer.step()
        CRITIC.optimizer.zero_grad()
    
        if epoch >0 and not epoch%(int(display_iter/n_agents)):
            duration_hist.append(np.average(np.array(duration_iter)))       
            loss_hist.append(np.average(np.array(loss_pg_iter)))
            loss_qv_hist.append(np.average(np.array(loss_qv_iter)))
            reward_history.append(np.average(np.array(reward_iter)))
            entropy_history.append( np.average(np.array(entropy_iter)) )
            
            ACTOR_diff = ACTOR.compare_weights(ACTOR_initial_weights)
            CRITIC_diff = CRITIC.compare_weights(CRITIC_initial_weights)
            
            print('')
            print(f'EPOCH {epoch}/{n_epochs}')
            
            print(f'ACTOR differences : {np.round(ACTOR_diff,5)}')
            print(f'CRITIC differences : {np.round(CRITIC_diff,5)}')
            
            ACTOR_initial_weights = deepcopy(ACTOR.state_dict())
            CRITIC_initial_weights = deepcopy(CRITIC.state_dict())
    
            print(f'last {display_iter} averages')
            
            print(f'loss Actor  : {np.round(np.average(np.array(loss_pg_iter)))}')
            print(f'loss Critic : {np.round(np.average(np.array(loss_qv_iter)))}')
            print(f'entropy     : {np.round( 100*np.average(np.array(entropy_iter))/entropy_max,2)}%')
            print(f'cum reward  : {np.round( np.average(np.array(reward_iter)),2)}')
            print(f'av. duration: {np.round(np.average(np.array(duration_iter)),2)}')      
            
            reward_iter =[]
            loss_pg_iter = []
            loss_qv_iter = []
            duration_iter = []
            entropy_iter = []
    
    
    # save data to file
    if save_data:
        arrays_tup = (    np.array([loss_qv_hist]).T,
                          np.array([loss_hist]).T,
                          np.array([entropy_history]).T,
                          np.array([reward_history]).T,
                          np.array([duration_hist]).T  )
        results_array = np.concatenate( arrays_tup, axis = 1)
        
        file_name = 'AC_test_' + str(n_agents) + '_agents.npy'
        with open(file_name, 'wb') as f:
            np.save(f, results_array)
    
    
    if generate_plots:
    
        sim_length = time.time()-t0
        #print(f'total duration: {np.round(sim_length,2)}s')
        
        fig0, ax0 = plt.subplots(3,1)    
        
        ax0[0].plot(loss_qv_hist)
        ax0[0].legend(['qv loss'])
        ax0[1].plot(loss_hist)
        ax0[1].legend(['pg loss'])
        ax0[2].plot(entropy_history)
        ax0[2].legend(['entropy'])
        
        fig0.suptitle(f'N agents: {n_agents}, sim length: {np.round(sim_length,2)}s', fontsize=14)
        
        fig, ax = plt.subplots(2,1)
        N = 100
        
        #ax[1].plot(reward_history)
        #ax[0].plot(np.zeros(len(reward_history)))
        ax[0].plot(reward_history)
        ax[0].legend(['average reward'])
        #ax[0].plot(np.convolve(reward_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])
        #ax[0].plot(0.5*np.ones(len(reward_history)))
        ax[1].plot(duration_hist)
        ax[1].legend(['average duration'])
        
        fig.suptitle(f'N agents: {n_agents}, sim length: {np.round(sim_length,2)}s', fontsize=14)
        
        
############################################################################


if __name__ == "__main__":
    
    env = main( n_agents = args.n_agents, generate_plots = args.generate_plots, \
               save_data = args.save_data, tot_iterations = args.tot_iterations )
        

        