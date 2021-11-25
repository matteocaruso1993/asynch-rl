# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:29:34 2020

@author: Enrico Regolin
"""

#%%

from .InvertedPendulum import InvertedPendulum
import gym
from gym import spaces
import numpy as np
import time



#%%

DEBUG = False

class CartPoleEnv(gym.Env):
    
    def __init__(self, sim_length_max = 30, difficulty = 0):
        # cartpole params
        
        self.env_type = 'CartPole'
        
        self.done_reward = [100, 20, 10]
        
        #self.weight = [0.1, 2, .25]  # tracking error, vertical angle, changes in control action
        self.weight = [0.05, 0.85, .1]  # tracking error, vertical angle, changes in control action
        
        self.sim_length_max = sim_length_max # max simulation length (in seconds)
        self.dt = 0.05
        self.cartpole = InvertedPendulum()
        
        self.difficulty = difficulty
        
        self.phi = np.array([0,0,0,0])
        
        if self.difficulty == 0: 
            self.max_distance = 2  # if actual distance is greater than this threshold, task has "failed"
            self.reference_speed_factor = .5
        elif self.difficulty == 1:
            self.max_distance = 1.5  # if actual distance is greater than this threshold, task has "failed"
            self.reference_speed_factor = .75
        elif self.difficulty == 2:
            self.max_distance = 1  # if actual distance is greater than this threshold, task has "failed"
            self.reference_speed_factor = 1
        elif self.difficulty == 3:
            self.max_distance = .9  # if actual distance is greater than this threshold, task has "failed"
            self.reference_speed_factor = 1.2
        
        # target params
        
        
        
        self.max_force = np.array([self.cartpole.force_max])
        self.max_state = np.append(self.cartpole.state_max_value, self.cartpole.state_max_value[0])

        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force) #, shape=(size__,))
        self.observation_space = spaces.Box(low=-self.max_state, high=self.max_state) # , shape=(len(state_max),))

        # initialize state
        self.state = np.zeros(5) # in gym-like env state has to include target
        self.duration = None

    #####################################################################################################            
    def get_max_iterations(self):
        return int(round(self.sim_length_max / self.dt))

    #####################################################################################################            
    def get_actions_structure(self):
        pass # implemented in wrapper

        
    #####################################################################################################            
    def get_observations_structure(self):
        return len(self.state)

    #####################################################################################################            
    # this function returns a traditional control input to be used instead of the ML one
    def get_controller_input(self, discretized = False, bins = 10):
        self.cartpole.computeControlSignals(self.state[:-1],self.state[-1])
        ctrl_signal = self.cartpole.get_ctrl_signal()
        if not discretized:
            return ctrl_signal
        else:
            ctrl_space = np.linspace(-self.max_force, self.max_force, bins+1)
            idx = np.argmin(np.abs(ctrl_space - ctrl_signal))
            return ctrl_space[idx]


    #####################################################################################################            
    def get_complete_sequence(self):
        if hasattr(self, 'stored_states_sequence'):
            return self.stored_states_sequence, self.stored_ctrl_sequence
        else:
            return None


    #####################################################################################################            
    def step(self,action, *args):

        
        info = {'outcome' : None}
        
        self.generate_target()
        if self.duration == 0:
            reset_run = True
        else:
            reset_run = False
        
        state_4 = self.cartpole.step_dt(self.dt, self.state[:-1], action, hold_integrate = True, reset_run =reset_run)
       
        self.state = np.append(state_4, self.x_target)
        self.cartpole.store_results(state_4, self.x_target, action)

        duration_prize = np.clip(2*self.duration/self.sim_length_max, 0, 1)
        #evaluate rewards
        if  np.abs(self.state[2]) > np.pi/2:
            done = True
            reward = -self.done_reward[0]*(1-0.5*duration_prize) # 50   # 
            info['termination'] = 'lost-balance'
        
        elif self.duration/self.sim_length_max > 1/4 and np.abs(self.state[0]-self.x_target) > self.max_distance:
            done = True
            reward = -self.done_reward[1]*(1 - 0.5*duration_prize + np.average(np.array(self.target_dist_hist)) )
            info['termination'] = 'lost-tracking'
            
        else:
            #print(f'current time = {self.duration}, max time = {self.sim_length_max}')
            target_penalty = np.abs(self.state[0]-self.state[4])/self.max_state[0]
            self.target_dist_hist.append(target_penalty)
            
            ctrl_penalty = ((np.diff(self.cartpole.ctrl_inputs[-2:])[-1]/(2*self.max_force))**2)[0]
            pole_angle_penalty = np.minimum(1,(3*self.state[2]/self.max_state[2])**(2) ) # np.minimum(1,(3*x)**2)
            
            
            if DEBUG:
                print('')
                print(f'action : {action}')
                print(f'target: {np.round(target_penalty,4)}, angle: {np.round(pole_angle_penalty,4)}, ctrl: {np.round(ctrl_penalty,4)}')
            # np.clip((10-self.duration),0,10) +
            reward =  1-( self.weight[0]*target_penalty + self.weight[1]*pole_angle_penalty + self.weight[2]*ctrl_penalty) #- control_sign_penalty
            #print(reward)
            # if this becomes negative there miht be an incentive to stop the simulation early!
            if self.duration >= self.sim_length_max:
                reward += self.done_reward[2] * (1 - np.average(np.array(self.target_dist_hist)) )
                done = True
                info['termination'] = 'success'
            else:
                done = False
        
        self.duration += self.dt
        
        if done:
            self.stored_states_sequence = np.append(self.cartpole.solution, self.cartpole.target_pos[:,np.newaxis], axis=1)
            self.stored_ctrl_sequence = self.cartpole.ctrl_inputs
        
            if DEBUG:
                print(f'final reward fallen pole: {reward}')
                print(f'duration/sim_max : {self.duration/self.sim_length_max}') 

        #time.sleep(0.5)
        #print(self.state/self.max_state)
        return self.state/self.max_state, reward, done, info


    #####################################################################################################
    def generate_target(self):
        
        omega_1 = .3*self.reference_speed_factor
        omega_2 = .1*self.reference_speed_factor
        omega_3 = .2*self.reference_speed_factor
        omega_4 = .4*self.reference_speed_factor

        if self.duration == 0:
            phi_1 = np.pi/4*(np.random.random()-0.5)
            phi_2 = np.pi/4*(np.random.random()-0.5)
            phi_3 = np.pi/4*(np.random.random()-0.5)
            phi_4 = np.pi/4*(np.random.random()-0.5)
            self.phi = np.array([phi_1, phi_2, phi_3, phi_4])
        
        def fun_val(t):
            signal = 2*np.sin(omega_1*t+ self.phi[0]) + 0.5*np.sin(omega_2*t + np.pi/7+ self.phi[1]) + \
                0.8*np.sin(omega_3*t - np.pi/12+ self.phi[2])+ 1*np.cos(omega_4*t+np.pi/5+ self.phi[3])
            return signal
    
        alpha = 0.01
        self.x_target = fun_val(self.duration) - fun_val(0)*np.exp(-alpha*self.duration)


        
    #####################################################################################################
    def reset(self, save_history = False, evaluate = False):
                
        self.duration = 0
        self.x_target = 0
        self.target_dist_hist = []
        
        if evaluate:
            self.state = np.array([0,0,0.05*np.random.randn(),0,0])
        else:
            self.state =self.max_state*0.5*(2*np.random.rand(5)-1)

        self.reset_update_storage(save_history = save_history)
        
        return self.state/self.max_state

    #####################################################################################################
    def reset_update_storage(self, save_history = False):
        self.cartpole.reset_store(self.state[:-1], ml_ctrl = True, save_history = save_history)


    #####################################################################################################
    def plot_graphs(self):
        time_line = np.arange(0.0, self.duration, self.dt)
        self.cartpole.plot_graphs(time_line, save = True, no_norm = False, ml_ctrl = True)


    #####################################################################################################    
    # currently copy/paste from robot-env
    def render(self,mode = 'human',iter_params = (0,0,0), action = 0):

        #self.cartpole.store_results(self.state[:-1], self.state[-1], action) # already stored during simulation
        self.cartpole.store_render_data(iter_params)
        
    #####################################################################################################    
    # currently copy/paste from robot-env
    def get_gif(self, fig, save_history = False):
        self.reset_update_storage(save_history= save_history)
        return self.cartpole.generate_gif(self.dt, sample_step=1, fig = fig)


#%%
"""

if __name__ == "__main__":
    
    continue_training = False
    
    # hyperparams
    tot_iter = 15000    
    
    gamma = 0.95
    beta = 0.1

    loss_qv_i = 0
    loss_pg_i = 0

    n_iter = 200 # display

    n_actions = 17
    
    prob_even = 1/n_actions*torch.ones(n_actions)
    entropy_max = np.round(torch.sum(-torch.log(prob_even)*prob_even).item(),3)
    game = CartPoleEnv()
    action_pool = np.linspace(-1,1,n_actions)*game.max_force

    if not continue_training:
    
        actor =  LinearModel(0, 'LinearModel0', 0.001, n_actions , 5, *[50,50], softmax=True ) 
        critic = LinearModel(1, 'LinearModel1', 0.001, 1, 5, *[50,50] ) 
    
        reward_history =[]
        loss_hist = []
        loss_qv_hist = []
        duration_hist = []
        entropy_history = []
    
    for iteration in tqdm(range(tot_iter)):
        
        done = False
        
        state = game.reset()
        
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
            advantage_loss += (  R - critic( state_sequences[-1-i].float()) )**2          
 
        total_loss = loss_policy + advantage_loss
     
        total_loss.backward()
        actor.optimizer.step()
        critic.optimizer.step()
 
        duration_hist.append(len(traj_rewards))       
 
        loss_pg_i = np.round(loss_policy.item(),3)
        loss_qv_i = np.round(advantage_loss.item(),3)

        loss_hist.append(loss_pg_i)
        loss_qv_hist.append(loss_qv_i)

        reward_history.append(cum_reward)
        entropy_history.append( round(torch.mean(torch.cat([t.unsqueeze(0) for t in traj_entropy])).item(),2) )

        if iteration> 0 and not iteration%n_iter:
            print(f'last {n_iter} averages')
            
            print(f'loss Actor {round( sum(abs(np.array(loss_hist[-n_iter:])))/n_iter )}')
            print(f'loss Critic {round( sum(loss_qv_hist[-n_iter:])/n_iter )}')
            print(f'entropy {round( 100*sum(entropy_history[-n_iter:])/entropy_max/n_iter,2)}%')
            print(f'cum reward {round( sum(reward_history[-n_iter:])/n_iter , 2)}')
            print(f'duration {round(sum( duration_hist[-n_iter:]) / n_iter,2 )}')        

    fig0, ax0 = plt.subplots(3,1)
    
    ax0[0].plot(loss_qv_hist)
    ax0[1].plot(loss_hist)
    ax0[2].plot(entropy_history)


    fig, ax = plt.subplots(2,1)
    N = 100

    #ax[1].plot(reward_history)
    #ax[0].plot(np.zeros(len(reward_history)))
    ax[0].plot(reward_history)
    #ax[0].plot(np.convolve(reward_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])
    #ax[0].plot(0.5*np.ones(len(reward_history)))
    ax[1].plot(duration_hist)
"""




#%%

"""

if __name__ == "__main__":
    
    continue_training = False
    
    # hyperparams
    tot_iter = 15000    
    
    gamma = 0.95
    beta = 0.1

    loss_qv_i = 0
    loss_pg_i = 0


    n_iter = 200 # display

    n_actions = 3
    
    prob_even = 1/n_actions*torch.ones(n_actions)
    entropy_max = np.round(torch.sum(-torch.log(prob_even)*prob_even).item(),3)
    game = CartPoleEnv()
    action_pool = np.linspace(-1,1,n_actions)*game.max_force

    if not continue_training:
    
        actor =  LinearModel(0, 'LinearModel0', 0.001, n_actions , 5, *[10,10], softmax=True ) 
        critic = LinearModel(1, 'LinearModel1', 0.001, 1, 5, *[10,10] ) 
    
        reward_history =[]
        loss_hist = []
        loss_qv_hist = []
        duration_hist = []
        entropy_history = []
    
    for iteration in tqdm(range(tot_iter)):
        
        done = False
        
        state = game.reset()
        
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
            advantage_loss += (  R - critic( state_sequences[-1-i].float()) )**2          
 
        total_loss = loss_policy + advantage_loss
     
        total_loss.backward()
        actor.optimizer.step()
        critic.optimizer.step()
 
        duration_hist.append(len(traj_rewards))       
 
        loss_pg_i = np.round(loss_policy.item(),3)
        loss_qv_i = np.round(advantage_loss.item(),3)

        loss_hist.append(loss_pg_i)
        loss_qv_hist.append(loss_qv_i)

        reward_history.append(cum_reward)
        entropy_history.append( round(torch.mean(torch.cat([t.unsqueeze(0) for t in traj_entropy])).item(),2) )

        if iteration> 0 and not iteration%n_iter:
            print(f'last {n_iter} averages')
            
            print(f'loss Actor {round( sum(abs(np.array(loss_hist[-n_iter:])))/n_iter )}')
            print(f'loss Critic {round( sum(loss_qv_hist[-n_iter:])/n_iter )}')
            print(f'entropy {round( 100*sum(entropy_history[-n_iter:])/entropy_max/n_iter,2)}%')
            print(f'cum reward {round( sum(reward_history[-n_iter:])/n_iter , 2)}')
            print(f'duration {round(sum( duration_hist[-n_iter:]) / n_iter,2 )}')        

    fig0, ax0 = plt.subplots(3,1)
    
    ax0[0].plot(loss_qv_hist)
    ax0[1].plot(loss_hist)
    ax0[2].plot(entropy_history)


    fig, ax = plt.subplots(2,1)
    N = 100

    #ax[1].plot(reward_history)
    #ax[0].plot(np.zeros(len(reward_history)))
    ax[0].plot(reward_history)
    #ax[0].plot(np.convolve(reward_history, np.ones(N)/N, mode='full')[int(N/2):-int(N/2)+1])
    #ax[0].plot(0.5*np.ones(len(reward_history)))
    ax[1].plot(duration_hist)

"""