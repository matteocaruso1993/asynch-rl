#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:33:46 2021

@author: rodpod21
"""

#
n_agents_vect = [1,2,4,5,8,10,20]
results = {}

for n_agents in n_agents_vect:
    file_name = 'AC_test_' + str(n_agents) + '_agents.npy'
    with open(file_name, 'rb') as f:
        results[str(n_agents)] = np.load(f)


output_arrays = {}
for i in range(results['5'].shape[1]):
    # iterate over the output result
    
    output_arrays[i] = np.zeros((results['5'].shape[0], len(n_agents_vect)))
    
    j = 0
    for k,result in results.items():
        output_arrays[i][:,j] = result[:,i]
        j += 1
    
#%%


fig0, ax0 = plt.subplots(3,1)    

ax0[0].plot(output_arrays[0])
ax0[0].set_ylabel('qv loss')
ax0[0].legend(n_agents_vect)

ax0[1].plot(output_arrays[0])
ax0[1].set_ylabel('pg loss')
ax0[1].legend(n_agents_vect)

ax0[2].plot(output_arrays[2])
ax0[2].set_ylabel('entropy')
ax0[2].legend(n_agents_vect)


fig, ax = plt.subplots(2,1)

#ax[1].plot(reward_history)
#ax[0].plot(np.zeros(len(reward_history)))
ax[0].plot(output_arrays[3])
ax[0].set_ylabel('average reward')
ax[0].legend(n_agents_vect)

#ax[0].plot(np.convolve(reward_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])
#ax[0].plot(0.5*np.ones(len(reward_history)))
ax[1].plot(output_arrays[4])
ax[1].set_ylabel('average duration')
ax[1].legend(n_agents_vect)


