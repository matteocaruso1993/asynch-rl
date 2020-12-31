import gym
import numpy as np

from gym.spaces import Box, Discrete

################################
################################
class DiscretizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, Box)

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        low = np.array(low)
        high = np.array(high)

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in
                         zip(low.flatten(), high.flatten())]
        self.ob_shape = self.observation_space.shape

        print("New ob space:", Discrete((n_bins + 1) ** len(low)))
        self.observation_space = Discrete(n_bins ** len(low)) 

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.val_bins)]
        return self._convert_to_one_number(digits)


################################
################################
class DiscretizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_bins_act=10, low_act=None, high_act=None):
        super().__init__(env)
        assert isinstance(env.action_space, Box)

        low_act = self.action_space.low if low_act is None else low_act
        high_act = self.action_space.high if high_act is None else high_act
        
        if not hasattr(low_act, '__len__'):
            low_act = [low_act]
            high_act = [high_act]
            
        if not isinstance(low_act, np.ndarray):
            low_act = np.array(low_act)
            high_act = np.array(high_act)
        
        self.act_space_dim = len(low_act)
        
        if self.act_space_dim >1:
            
            if not isinstance(n_bins_act, list):
                self.n_bins_act = []
                [self.n_bins_act.append(n_bins_act) for i in range(self.act_space_dim) ]
            else:
                self.n_bins_act = n_bins_act
                
            self.val_bins_act = [np.linspace(l, h, self.n_bins_act[i] + 1) for i,(l, h) in
                             enumerate(zip(low_act.flatten(), high_act.flatten()))]
            
        elif self.act_space_dim  == 1:
            self.val_bins_act = np.linspace(low_act[0], high_act[0], self.n_bins_act + 1)

        self.act_shape = self.action_space.shape

        print("New act space:", Discrete( self.get_actions_structure() ) )
        self.action_space = Discrete( self.get_actions_structure() )
        
        self.act_2D_flattened = None
        if self.act_space_dim==1:
            self.act_1D = self.val_bins_act
            
        elif self.act_space_dim==2:
            xv, yv = np.meshgrid(self.val_bins_act[0], self.val_bins_act[1])
            self.act_2D_flattened = np.stack((xv.flatten(),yv.flatten()  ))
            
        elif self.act_space_dim>2:
            arg = [self.val_bins_act[i] for i in range(self.act_space_dim)]
            xx_out = np.meshgrid(*arg)
            self.act_nD_flattened = np.stack(([xx.flatten() for xx in xx_out ]))

    ###################################################    
    def get_actions_structure(self):
        return np.prod(np.array(self.n_bins_act)+1)   # n_actions depends strictly on the type of environment
    
    ###################################################
    def step(self, action, random_gen = False):
        return self.env.step(action, random_gen)
    
    def action_2D(self,action_bool_array):
        return self.step(self.act_2D_flattened[:,np.where(action_bool_array)].reshape(2,))            

    def action_nD(self,action_bool_array, random_gen = False):
        return self.step(self.act_nD_flattened[:,np.where(action_bool_array)].reshape(self.act_space_dim,), random_gen)            

    def action(self, action_bool_array, random_gen = False):
        #if self.act_2D_flattened is not None:
        if self.act_space_dim  == 1 :
            return self.step(self.act_1D[np.where(action_bool_array)[0]])
        elif self.act_space_dim  == 2 : 
            return self.action_2D(action_bool_array)
        elif self.act_space_dim > 2 : 
            return self.action_nD(action_bool_array, random_gen)
        
        raise NotImplementedError
        
    ###############################################
    # this function returns a traditional control input to be used instead of the ML one
    def get_control_idx(self, discretized = True):
        u_ctrl = self.env.get_controller_input(discretized = discretized, bins = self.n_bins_act)
        return self.get_action_idx(u_ctrl)
        
    def get_action_idx(self, action):
        if self.act_space_dim  == 1:
            action_i = np.abs(self.val_bins_act - action).argmin()
        elif self.act_space_dim  == 2:
            action_i = np.linalg.norm((self.act_2D_flattened.T - action), axis = 1).argmin()
        elif self.act_space_dim  > 2:
            action_i = np.linalg.norm((self.act_nD_flattened.T - action), axis = 1).argmin()
        return action_i
    
    
################################
################################
class ContinuousHybridActionWrapper(gym.ActionWrapper):
    # for Net outputs in the [0,1] range, it applies to the environment:
    # - a continuously varying input between low_act and high act for actions which index in action structure == 0
    # - an int n: 0<=n<=N for actions which index in action structure == N
    
    
    def __init__(self, env,action_structure = None, low_act=None, high_act=None):
        super().__init__(env)
        assert isinstance(env.action_space, Box)
        
        self.low_act = self.action_space.low if low_act is None else low_act
        self.high_act = self.action_space.high if high_act is None else high_act
        
        self.delta_act = self.high_act - self.low_act

        self.action_structure = [ 0 for _ in range(len(self.low_act))]
        if action_structure is not None:
             self.action_structure = [np.linspace(0,1,n_act+1) if n_act>0 else 0 for i, n_act in enumerate(action_structure) ]
        
        self.n_actions = len(self.action_structure)
             
    ###################################################    
    def get_actions_structure(self):
        return self.n_actions
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action, source = None):
        # here I have to convert the action requested by the net to the one received by the enviroment
        if len(action)>1:
            action_out = [ self.low_act[i] + act*self.delta_act[i] if self.action_structure[i]==0 \
                          else  np.abs(self.action_structure[i] - act).argmin() \
                          for i,act in enumerate(action)]
            return action_out
            
        else:
            raise NotImplementedError

    def reverse_action(self, action):
        raise NotImplementedError