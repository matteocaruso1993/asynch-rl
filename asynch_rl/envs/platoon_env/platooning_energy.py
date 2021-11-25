#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:24:36 2020

@author: Enrico Regolin
"""



import numpy as np
from scipy.interpolate import griddata, interpolate 
import matplotlib.pyplot as plt
import torch

import os

DEBUG = False


#%%

class ElMotor():
    
    def __init__(self, max_speed = 1200):
        
        self.max_speed = max_speed
        self.max_torque = 180
        
        self.tq_margin = 5
        
        self.EM_w_list=np.array([0,95,190,285,380,475,570,665,760,855,950,1045,1140, 1200])
        self.EM_T_list=np.array([0,11.25,22.5,33.75,45,56.25,67.5,78.75,90,101.25,112.5,123.75,135,146.25,157.5,168.75,180])
        
        x2d, y2d = np.meshgrid(self.EM_w_list, self.EM_T_list)
       
        self.x2d = x2d
        self.y2d = y2d
       
        self.x_speed_flat = x2d.flatten()
        self.y_torque_flat = y2d.flatten()
       
        #self.EM_T_max_list   = np.array([179.1,179,180.05,180,174.76,174.76,165.13,147.78,147.78,109.68,109.68,84.46,84.46])
        self.EM_T_max_list   = np.array([180,180,180,180,174.76,170,165.13,150,137.78,115.68,105.68,94.46,84.46, 78])
        
        self.f_max_rq = interpolate.interp1d(self.EM_w_list, self.EM_T_max_list, kind =  "cubic", fill_value="extrapolate")

        self.efficiency = np.array([
        [.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50],
        [.68,.70,.71,.71,.71,.71,.70,.70,.69,.69,.69,.68,.67,.67,.67,.67,.67],
        [.68,.75,.80,.81,.81,.81,.81,.81,.81,.81,.80,.80,.79,.78,.77,.76,.76],
        [.68,.77,.81,.85,.85,.85,.85,.85,.85,.84,.84,.83,.83,.82,.82,.80,.79],
        [.68,.78,.82,.87,.88,.88,.88,.88,.88,.87,.87,.86,.86,.85,.84,.83,.83],
        [.68,.78,.82,.88,.88,.89,.89,.89,.88,.88,.87,.85,.85,.84,.84,.84,.83],
        [.69,.78,.83,.87,.88,.89,.89,.88,.87,.85,.85,.84,.84,.84,.84,.84,.83],
        [.69,.73,.82,.86,.87,.88,.87,.86,.85,.84,.84,.84,.84,.84,.84,.84,.83],
        [.69,.71,.80,.83,.85,.86,.85,.85,.84,.84,.84,.84,.84,.84,.83,.83,.83],
        [.69,.69,.79,.82,.84,.84,.84,.84,.83,.83,.83,.83,.83,.83,.83,.82,.82],
        [.69,.68,.75,.81,.82,.81,.81,.81,.81,.81,.81,.80,.80,.80,.80,.80,.80],
        [.69,.68,.73,.80,.81,.80,.76,.76,.76,.76,.76,.76,.76,.76,.75,.75,.75],
        [.69,.68,.71,.75,.75,.75,.75,.75,.75,.75,.75,.75,.74,.74,.74,.74,.74],
        (0.9*np.array([.69,.68,.71,.75,.75,.75,.75,.75,.75,.75,.75,.75,.74,.74,.74,.74,.74])).tolist() ]).T
       
        self.efficiency_flat = self.efficiency.flatten()
        self.get_eff_matrix()
       
        
    def getEfficiency(self, speed, torque): 
        if torch.is_tensor(torque):
            torque = torque.item()
        if torch.is_tensor(speed):
            speed = speed.item()
        
        points = (self.x_speed_flat, self.y_torque_flat)
        pair = (np.abs(speed), np.abs(torque))
        grid = griddata(points, self.efficiency_flat, pair, method = "cubic")
        # todo: debug
        grid[np.abs(torque) > self.f_max_rq(np.abs(speed)) + self.tq_margin ] = np.nan
        
        # print(grid)
        return grid
   
    
    def getMaxTorque(self, speed, return_tensor = True):
        #max_tq = np.interp(np.abs(speed.cpu().detach().numpy()), self.EM_w_list, self.EM_T_max_list)
        if torch.is_tensor(speed):
            max_tq = self.f_max_rq(np.abs(speed.cpu().detach().numpy()))
        else:
            max_tq = self.f_max_rq(np.abs(speed))
        if isinstance(max_tq, np.ndarray) and max_tq.shape:
            if return_tensor:
                return torch.tensor(max_tq[0])
            else:
                return max_tq[0]
        else:
            if return_tensor:
                return torch.tensor(max_tq)
            else:
                return max_tq


    def getMaxPower(self):
        if not hasattr(self, 'speed_vect'):
            self.speed_vect = np.linspace(0,self.max_speed,201)
        return np.amax(self.f_max_rq(self.speed_vect)*self.speed_vect)
    

    def get_eff_matrix(self):
        
        self.speed_vect = np.linspace(0,self.max_speed,201)
        self.torque_vect = np.linspace(0,self.max_torque,151)
        xx, yy = np.meshgrid(self.speed_vect, self.torque_vect)

        self.eff_matrix = self.getEfficiency(xx, yy) #.reshape((emot.speed_vect.shape[0],emot.torque_vect.shape[0]))
                
        self.eff_matrix[yy >  self.f_max_rq(xx) ] = np.nan


    def plotEffMap(self, scatter_array = None):
        
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.set_xlim([0,self.max_speed])
        ax1.set_ylim([0,self.max_torque])
       
        #ax = plt.gca()
        ax1.set_aspect(3.5)
        
        levels = np.linspace(0.5, 0.9, 25)
        
        ax1 = plt.contourf(self.speed_vect, self.torque_vect, self.eff_matrix,levels = levels ,cmap = 'jet')

        if scatter_array is not None:
            plt.scatter(scatter_array[:,0],scatter_array[:,1], s = 10, c = 'k')

        plt.plot(self.speed_vect, self.f_max_rq(self.speed_vect) , 'k')
        plt.xlim([0,1200])
        plt.ylim([0,200])
        
        plt.xlabel('$n_m$ (rpm)')
        plt.ylabel('$T_m$ (Nm)')
        #plt.plot(self.EM_w_list,self.EM_T_max_list , 'k')

        cbar =plt.colorbar(ax1, cax = fig1.add_axes([0.78, 0.52, 0.03, 0.3]))
        cbar.ax.locator_params(nbins=5)
        #plt.show()
        return fig1


    def save_tq_limit_torch(self):
        torque_limit = torch.tensor(np.concatenate( (self.speed_vect[:,np.newaxis], self.f_max_rq(self.speed_vect)[:,np.newaxis]), axis = 1 ))
        
        file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'torque_limit.pt')
        torch.save(torque_limit, file_name)

#%%

"""
emot = ElMotor(max_speed = 1200)
fig = emot.plotEffMap()
#ax = plt.gca()

fig.savefig('efficiency_map.eps')
"""


#%%
class Car():
    """ Describes the physical behaviour of the vehicle """
    def __init__(self, initial_speed = 0.0, max_acceleration = 5, n_gears = 1):
        #self.device=device
        self.n_gears = n_gears

        self.gravity = 9.81 #m/s^2
        self.position = 0
        self.velocity = initial_speed
        self.acceleration = 0
        self.friction_coefficient = 0.01 # will be ignored

        self.mass = 800       #kg
        self.rho =  1.22          #the air density, 
        self.aer_coeff = 0.4     #the aerodynamic coefficient
        self.veh_surface  = 2  #equivalent vehicle surface
        self.rr_coeff =  8e-3     #rolling resistance coefficient
        
        if self.n_gears == 1:
            self.gear_ratio = [10]
        elif self.n_gears == 2:
            self.gear_ratio = [12, 7]
        elif self.n_gears == 3:
            self.gear_ratio = [14, 10, 7]
        
        self.wheel_radius = 0.3  #effective wheel radius
        self._max_whl_brk_torque = 4000  #Nm

        self.e_motor = ElMotor()

        self._max_acceleration = max_acceleration   #m/s^2
        self._min_acceleration = -self._max_acceleration

        self._max_velocity =  0.95 * self.e_motor.max_speed / np.array(self.gear_ratio) * self.wheel_radius
        
        self._min_velocity = 0.0

        self.max_br_torque = 2000
        self.max_e_tq = self.e_motor.max_torque
        self.min_e_tq = - self.max_e_tq
        self.e_motor_speed = 0
        self.e_torque= 0
        self.br_torque= 0
        self.e_power = 0
        self.eff = 0
        
        self.current_gear = 0

    def motor_efficiency(self):
        self.eff = self.e_motor.getEfficiency(self.e_motor_speed, self.e_torque)

    
    def calculate_wheels_torque(self, e_torque, br_torque ):
        self.br_torque = np.clip(br_torque, 0, self._max_whl_brk_torque)
        self.e_torque = np.clip(e_torque, self.min_e_tq, self.max_e_tq)
        return self.e_torque*self.gear_ratio[self.current_gear] - self.br_torque

    def resistance_force(self):
        F_loss = 0.5*self.rho*self.veh_surface*self.aer_coeff*(self.velocity**2) + \
            self.rr_coeff*self.mass*self.gravity*self.velocity
        return F_loss


    def update(self, dt, norm_e_torque, norm_br_torque, gear_n = 0):
        
        if self.n_gears > 1:
            self.current_gear = int(gear_n)
            #print(self._max_velocity[self.current_gear])
        
        #Differential equation for updating the state of the car
        if torch.is_tensor(norm_e_torque):
            norm_e_torque = norm_e_torque.item()
        if torch.is_tensor(norm_br_torque):
            norm_br_torque = norm_br_torque.item()        

        in_wheels_torque = self.calculate_wheels_torque(np.clip(norm_e_torque, -1,1)*self.max_e_tq, \
                                                        np.clip(norm_br_torque, 0, 1)*self.max_br_torque)

        acceleration = (in_wheels_torque/self.wheel_radius - self.resistance_force() ) / self.mass
           
        self.acceleration = np.clip(acceleration, self._min_acceleration, self._max_acceleration)
        
        # self.velocity = torch.clamp(self.velocity + self.acceleration * dt, self._min_velocity, self._max_velocity)
        velocity = self.velocity + self.acceleration * dt
        self.velocity = np.clip(velocity, 0 ,self._max_velocity[self.current_gear] )
        
        self.e_motor_speed = self.velocity*self.gear_ratio[self.current_gear]/self.wheel_radius
        
        # update power consumed
        self.motor_efficiency()

        # check for NaN problem
        count = 0
        n_tentatives = 20
        while np.isnan(self.eff) and count < n_tentatives:
            self.e_torque = 0.95*np.abs(self.e_torque)*np.sign(self.e_torque)
            self.motor_efficiency()
            count +=1
        
        if torch.is_tensor(self.e_torque):
            self.e_torque = self.e_torque.item()
        effective_efficiency = self.eff**(-np.sign(self.e_torque))
        
        self.e_power = (self.e_motor_speed*self.e_torque)*effective_efficiency
        self.position += self.velocity * dt

        #update min/max e-torque based on new motor speed
        self.max_e_tq = self.e_motor.getMaxTorque(self.e_motor_speed, return_tensor=False)
        self.min_e_tq = -self.max_e_tq
        
      

        
        

#%%
if __name__ == "__main__":
    
    #%%
    journey_length = 200
    
    initial_speed = 10
    ref_vehicle_speed = 25 #m/s
    target_distance = 40
    ref_vehicle_position = 40 #m ahead of controlled vehicle
    dt = 1 #s
    cum_error = 0
    cum_energy = 0
    
    car = Car( initial_speed = initial_speed, change_gears= True)
    
    car.e_motor.plotEffMap()
    car.e_motor.getMaxPower()
    #%%
    
    # car position, car speed, vehicle position, vehicle speed
    state_storage = np.array([[0, initial_speed, ref_vehicle_position,ref_vehicle_speed]])
    # e torque, br torque
    ctrl_storage = np.array([[0, 0, 0, 0, cum_energy]])
    
    for i in range(journey_length):
        
        distance = ref_vehicle_position - car.position
        error = target_distance - distance
        cum_error += error
        
        norm_e_torque = np.clip(-(0.025*error +.001* cum_error),-1 , 1)
        
        norm_br_torque = 0
        if error > 20:
            norm_br_torque = 1
        
        if car.change_gears:
            if car.current_gear == 0 and car.velocity > 27:
                gear_n = 1
            elif car.current_gear == 1 and car.velocity < 25:
                gear_n = 0
            else:
                gear_n = car.current_gear
        else:
                gear_n = 0
                
        car.update(dt, norm_e_torque, norm_br_torque, gear_n)
    
        e_torque = norm_e_torque*car.max_e_tq*car.gear_ratio[car.current_gear]
        br_torque = -norm_br_torque*car.max_br_torque
    
        acc = 2*np.random.randn()
        ref_vehicle_speed += acc * dt
        ref_vehicle_speed = np.clip(ref_vehicle_speed, 5, 30)
        ref_vehicle_position += ref_vehicle_speed* dt
        
        power = car.e_power
        cum_energy += power*dt
        
        state_storage = np.append(state_storage, np.array([[car.position, car.velocity,ref_vehicle_position, ref_vehicle_speed ]]), axis = 0)
        ctrl_storage = np.append(ctrl_storage, np.array([[e_torque, br_torque, error, power, cum_energy]]), axis = 0)
        
        print(f'iteration {i}')
    
    
    fig1 = plt.figure()
    ax_0 = fig1.add_subplot(3,1,1)
    ax_1 = fig1.add_subplot(3,1,2)
    ax_2 = fig1.add_subplot(3,1,3)
    
    
    ax_0.plot(ctrl_storage[:,2])
    ax_0.plot(0*np.ones((journey_length,1)), 'k',linewidth=0.5)
    ax_0.plot(35*np.ones((journey_length,1)), 'r')
    #ax_0.plot(-20*np.ones((journey_length,1)))
    ax_0.legend(['distancing error','reference','crash line' ])
    
    ax_1.plot(state_storage[:,2])
    ax_1.plot(state_storage[:,0])
    ax_1.legend(['leader pos','car position'])
    
    
    ax_2.plot(state_storage[:,3])
    ax_2.plot(state_storage[:,1])
    ax_2.legend(['leader vel','car vel'])
    
    fig1.savefig('state_signals.png')
    
    
    plt.show()
    
    fig2 = plt.figure()
    ax0 = fig2.add_subplot(3,1,1)
    ax1 = fig2.add_subplot(3,1,2)
    ax2 = fig2.add_subplot(3,1,3)
    
    ax0.plot(ctrl_storage[:,3])
    ax0.legend(['power'])
    
    ax1.plot(ctrl_storage[:,4])
    ax1.legend(['cum energy'])
    
    ax2.plot(ctrl_storage[:,0])
    ax2.plot(ctrl_storage[:,1])
    ax2.legend(['electric tq','brake tq'])
    
    fig2.savefig('ctrl_signals.png')
    
    plt.show()
    
    
