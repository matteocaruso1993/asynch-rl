# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
#import sys
import time
import gym
from gym import spaces
import torch


import colorama
from colorama import Back, Style
colorama.init()


class Chess(gym.Env):

    
    def __init__(self, use_NN = False, max_n_moves = 100, pause = 3, rewards = [100,50, 1, 1e6], print_out = True, act_nD_flattened = None):
        
        self.env_type = 'Chess'
        self.chessboard_backup  = self.next_mover = self.model = None
        self.initialize_board()

        # externally editable params
        self.print_out = print_out
        self.pause = 0
        if self.print_out:
            self.pause = pause
        self.use_NN = use_NN
        self.max_n_moves = max_n_moves
        self.rewards = rewards
        
        if self.use_NN:
            self.act_nD_flattened = act_nD_flattened
        
        self.action_space = spaces.Box(low=np.array([0,0,0,0]), high=np.array([7,7,7,7])) #, shape=(size__,))
        
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


    def initialize_board(self):
        """ board initialization """
        self.chessboard = np.zeros((8,8),dtype = np.int)
        # pawns own
        self.chessboard[1,:] = np.ones((1,8))
        self.chessboard[0,:] = np.array([5,4,3,9,10,3,4,5])
        # pawns other
        self.chessboard[6,:] = -np.ones((1,8))
        self.chessboard[7,:] = -np.array([5,4,3,9,10,3,4,5])


    def check_board(self, l, n):
        """ safe evaluation of chessboard position (in case out of board position is requested)"""
        if 0<=l<=7 and 0<=n<=7 :
                return self.chessboard[n,l]
        else:
            return 0
        

    def render(self, action = None, own = True):
        """ render chessboard in terminal"""
        if self.previous_chessboard_out is None or action is None:
            print('---------------')
            chessboard_out = np.flipud(self.chessboard).copy().tolist()
            for i,row in enumerate(chessboard_out):
                chessboard_out[i] = [sym_rep(x) for x in row]
                print(' '.join(chessboard_out[i]))
            print('---------------')
            self.previous_chessboard_out = chessboard_out
        else:        
            l,n,L,N = action
            print('---------------   ---------------')
            chessboard_out = np.flipud(self.chessboard).copy().tolist()
            for i,row in enumerate(chessboard_out):
                chessboard_out[i] = [sym_rep(x) for j,x in enumerate(row)]
                if i!=7-N and i!=7-n:
                    endcond = '\n' if i not in [1,2,5,6] else " "
                    print(' '.join(self.previous_chessboard_out[i]), end="   ", flush=True)
                    print(' '.join(chessboard_out[i]), end=endcond, flush=True)
                    
                    if endcond == " ":
                        self.print_lost_pieces(i) 
                else:
                    print(' '.join(self.previous_chessboard_out[i]), end="   ", flush=True)
                     
                    for j,x in enumerate(row):
                        endcond = '\n' if j == 7 and i not in [1,2,5,6] else " "
                        if (j==L and i==7-N) or (j == l and i==7-n ):
                            if own:
                                print(Back.GREEN + sym_rep(x) + Style.RESET_ALL, end=endcond, flush=True)
                            else:
                                print(Back.RED   + sym_rep(x) + Style.RESET_ALL, end=endcond, flush=True)
                        else:
                            print(sym_rep(x), end=endcond, flush=True)
                        
                        if j == 7:
                            self.print_lost_pieces(i)
            print('---------------   ---------------')
            self.previous_chessboard_out = chessboard_out
        time.sleep(self.pause)
        print('')
        print('')
        print('')
        print('')
        print('')
        
        
    def print_lost_pieces(self,i):
        """ prints lost pieces of each player next to board"""
        if i == 1:
            print('      ', end = ' ', flush=True)
            print(' '.join(self.lost_pieces['player']['pawns']))
        elif i == 2:
            print('      ', end = ' ', flush=True)
            print(' '.join(self.lost_pieces['player']['others']))

        if i == 5:
            print('      ', end = ' ', flush=True)
            print(' '.join(self.lost_pieces['opponent']['pawns']))
        elif i == 6:
            print('      ', end = ' ', flush=True)
            print(' '.join(self.lost_pieces['opponent']['others']))


    def king_position(self, own = True):
        """ returns king position """
        sign = 2*int(own)-1
        n,l = np.where(self.chessboard == 10*sign)
        if len(n)== 0:
            print('no king!')
            return False
        else:       
            return l,n 


    def evaluate_move(self, action, own = True, hypoth_opp = False):         
        """ evaluates feasibility of proposed piece move"""
        l,n,L,N = action
        
        if not valid_action_space(action) or (self.check_board(L,N)*self.check_board(l,n))>0 :
            valid_move = False
        else:
            piece_type = np.abs(self.chessboard[n,l])
            if  piece_type == 1:
                valid_move = self.evaluate_pawn(action, own, hypoth_opp)
            elif piece_type == 3:
                valid_move = self.evaluate_bishop(action, hypoth_opp)
            elif piece_type == 5:
                valid_move = self.evaluate_tower(action, hypoth_opp)
            elif piece_type == 9:
                valid_move = self.evaluate_tower(action, hypoth_opp) or self.evaluate_bishop(action, hypoth_opp)
            elif piece_type == 4:
                valid_move = self.evaluate_horse(action, hypoth_opp)
            elif piece_type == 10:
                valid_move = self.evaluate_king(action,own, hypoth_opp )
            else:
                raise('Undefined move')
        return valid_move


    def reset(self, **kwargs):
        """ resets board to initial condition """
        self.initialize_board()
        self.next_mover = None
        self.previous_chessboard_out = None
        if self.print_out:
            self.render()
        self.moves_counter = 0
        self.stall = None
        
        self.lost_pieces = {'player': {'pawns' : [], 'others' : []}, 'opponent': {'pawns' : [], 'others' : []}}
                            
        if np.random.random()>0.5:
            self.next_mover = 'player'
        else:
            self.next_mover = 'opponent'
            opp_action = self.get_opponent_action()
            self.apply_move(opp_action)
            self.next_mover = 'player'
            if self.print_out:
                self.render(opp_action, own = False)
        

    def revert_move(self):
        """ cancels tentative move"""
        self.chessboard = self.chessboard_backup
        self.chessboard_backup  = None


    def apply_move(self, action, tentative = False):
        """ applies move. if tentative it s done only for evaluation purposes"""
        l,n,L,N = action        
        own = (self.chessboard[n,l]>0)

        self.chessboard_backup = self.chessboard.copy()
        if not tentative and self.print_out: 
            print('---------------------------------------------------------------')
            print('---------------------------------------------------------------')
            if self.chessboard[N,L] != 0:
                if not own:
                    self.lost_pieces['player']['pawns'].append(sym_rep(self.chessboard[N,L])) if self.chessboard[N,L] == 1 else self.lost_pieces['player']['others'].append(sym_rep(self.chessboard[N,L]))
                else:
                    self.lost_pieces['opponent']['pawns'].append(sym_rep(self.chessboard[N,L])) if self.chessboard[N,L] == -1 else self.lost_pieces['opponent']['others'].append(sym_rep(self.chessboard[N,L]))
                print(f'                   Lost piece: {sym_rep(self.chessboard[N,L])}')
            print('')
        self.chessboard[N,L] = self.chessboard[n,l]
        self.chessboard[n,l] = 0
        self.pawn_to_queen()
        
        if not tentative:
            if self.check_check(own = not own) and self.print_out:
                if own:
                    print('player checks opponent!')
                else:
                    print('opponent checks player!')            
            self.moves_counter += 1
            if self.print_out:
                print(f'Previous board --- Move N. {self.moves_counter}')
        
        # following check might be removed in the future
        if self.king_position(own = not own):
            if self.check_check(own = own):
                if tentative:
                    return False
                else:
                    raise('Going to checked position!')
            return True
        elif tentative:
            return False
        else:
            raise('King disappeared')
        
        
    def is_valid_move(self, action, own = True):
        """ returns True if move is valid"""
        sign = 2*int(own)-1
        l,n,L,N = action
        if sign*self.chessboard[n,l] > 0 and self.evaluate_move(action, own = own):
            if self.validate_action(action, own = own) is not None:
                return True
        return False
        
    
    def pawn_to_queen(self):
        """ convert to queen pawns in last row """
        last_row_pawns_idx = np.where(np.abs(self.chessboard)==1 )
        pawn_rep = [ np.array([idx_row, last_row_pawns_idx[1][i] ] ) for i,idx_row in enumerate(last_row_pawns_idx[0]) if idx_row in [0,7] ]
        if bool(pawn_rep):
            self.chessboard[pawn_rep[0][0], pawn_rep[0][1]] *= 9
            

    def step(self,action, random_gen = False):
        """ single step, comprises application of """
        info = {}
        done = False
        
        if isinstance(action, np.ndarray):
            action = tuple( action.astype(np.int) )
        
        if random_gen and not self.is_valid_move(action):
            action = self.random_move(own = True)
            info['move changed'] = action        
        
        if self.is_valid_move(action):
            self.apply_move(action)
            self.next_mover = 'opponent'
            if self.print_out:
                self.render(action)
            reward = 0
            mate_check = self.check_mate(own = False)
            if mate_check:
                if mate_check == 1:
                    reward = self.rewards[0]
                    done = True
                    info['winner'] = 'player'
                elif mate_check == 2:
                    reward = 0
                    done = True
                    info['winner'] = 'draw'
            else:
                opp_action = self.get_opponent_action()
                self.apply_move(opp_action)
                self.next_mover = 'player'
                if self.print_out:
                    self.render(opp_action, own = False)
                mate_check = self.check_mate()
                if mate_check:
                    if mate_check == 1:
                        reward = -self.rewards[0]
                        done = True
                        info['winner'] = 'opponent'
                    elif mate_check == 2:
                        reward = -self.rewards[1]
                        done = True
                        winner = 'draw'
                        info['winner'] = 'draw'
        else:
            reward = -self.rewards[3]
            done = True
        return self.get_state(), reward, done, info 


    def get_state(self, own = True, torch_output = False):
        """ returns snapshot used by NN"""
        if own:
            state = self.chessboard
        else:
            state = - np.flipud(np.fliplr(self.chessboard))
        if torch_output:
            return torch.tensor(state.flatten()).float()
        return state.flatten()


    def get_NN_action(self):
        """  """
        net_output = self.model(self.get_state(own = False, torch_output = True))
        
        action_bool_array = torch.zeros([8**4], dtype=torch.float32)
        action_index = torch.argmax(net_output)
        action_bool_array[action_index] = 1        
        
        return tuple(self.act_nD_flattened[:,np.where(action_bool_array)].reshape(4,).astype(np.int))
        
    

    def get_opponent_action(self):
        """ picks opponent action (random or NN based). If NN one is invalid, returns random"""
        if self.use_NN:
            action = self.get_NN_action()
            if self.is_valid_move(action, own = False):
                return action
        if self.use_NN and self.print_out:
            print('NN returned invalid move. Random one is generated!')
        return self.random_move(own = False)


    def check_if_kings_exist(self):
        """ debug function only """
        if not (self.king_position(own = False) and self.king_position(own = True)):
            raise('King disappeared')


    def random_move(self, own = False):
        """ randomly select action"""
        # get out of check if needed
        if self.check_check(own = own):
            for i in range(20):
                act = self.king_position(own=own)
                if act:
                    action = self.extract_move(*act, own = own )
                    value = self.validate_action(action, own)
                    if value is not None:
                        return action

        # try to check the opponent's king or grab pieces
        if self.moves_counter > 3:
            max_value = -0.5
            best_action = None
            for i in range(200):
                piece = self.extract_piece(own = own)
                action = self.extract_move(*piece, own = own)
                value = self.validate_action(action, own)
                if value is not None:
                    if self.check_check(own = not own):
                        value += 5.5
                    value -= self.loss_risk_value(action, own)

                    if value > max_value:
                        max_value = value
                        best_action = action
    
            if best_action is not None:
                return best_action

        # take first available random move                                      
        for i in range(1000):
            piece = self.extract_piece(own = own)
            action = self.extract_move(*piece, own = own )
            value = self.validate_action(action, own)
            if value is not None:
                return action
        
        raise('Solution not found!')
    
    
    def loss_risk_value(self, action, own, prob = 0.8):
        """ calculated value loss related to risk of losing moved piece"""
        l,n,L,N = action
        return prob*int(self.checked_position(L, N, own = own))*np.abs(self.chessboard[n,l])
    
    
    def validate_action(self,action, own):
        """ decide if action is valid (respects rules and doesn't put own king in check')"""
        if type(action) is tuple and len(action) == 4:
            if self.evaluate_move(action, own = own) and valid_action_space(action):
                valid, value = self.check_proof_action(action, own = own)
                if valid:
                    return value
        return None
    
    
    def extract_piece(self, own = True):
        """ select randomly a piece to move """
        sign = 2*int(own)-1
        nn,ll = np.where( sign*self.chessboard > 0 )
            
        idx = np.random.randint(len(nn))
        return ll[idx], nn[idx]
         
     
    def extract_move(self, l, n, own = True):
        """ select randomly a move for the given piece"""
        piece_idx = np.abs(self.check_board(l,n))
        moves = possible_moves(piece_idx, l, n, own)
        idx = np.random.randint(moves.shape[0])
        L,N = moves[idx, 0], moves[idx, 1]
        if not np.isscalar(L):
            L = L[0]
            N = N[0]
        return (l,n,L,N)

    
    def check_mate(self, own = True):
        """checkmate proof. if 1 checkmate, 2 stalemate, else 0"""
        # checks if we lost the game
        if not self.systematic_check(own):
            if self.check_check(own = own):
                return 1
            else:
                return 2
        elif self.moves_counter >= self.max_n_moves:
            return 2
        return False


    def systematic_check(self, own):
        """ checks if it possible to make a move"""
        sign = 2*int(own)-1
        pieces_pos = np.where(sign*self.chessboard > 0)
        for n, l in zip(pieces_pos[0], pieces_pos[1]):
            piece_idx = np.abs(self.check_board(l,n))
            for moves in possible_moves(piece_idx, l, n, own=own ):
                action = (l,n,moves[0], moves[1])
                if self.evaluate_move(action, own = own, hypoth_opp = False):
                    if self.check_proof_action(action)[0]:
                        return True
        return False

    
    def check_check(self, own = True):
        """ check if player's (own) king is under check"""
        if own:
            return self.checked_position( *self.king_position() )
        else:
            return self.checked_position( *self.king_position(own = False), own = False)

                                     
    def check_proof_action(self, action, own = True):
        """ checks that move doesn't lead to check. if own is False checks an opponent move"""
        sign = 2*int(own)-1
        move_value = None
        if self.apply_move(action, tentative = True):        
            check_free = not self.check_check(own = own)
            move_value = sign*(self.chessboard.sum()-self.chessboard_backup.sum())
            self.revert_move()
        else:
            self.revert_move()
            check_free = False
            
        return check_free, move_value
                                     

    def checked_position(self, L, N, own = True):
        """ check if given position is checked by opponent """
        if own:
            opponents_pos = np.where(self.chessboard < 0)
        else:
            opponents_pos = np.where(self.chessboard > 0)
        for n , l in zip( opponents_pos[0], opponents_pos[1] ):
            if not np.isscalar(l):
                l = l[0]
                n = n[0]
            if not np.isscalar(L):
                L = L[0]
                N = N[0]
            action = (l,n,L,N) 
            if self.evaluate_move(action, own = not own, hypoth_opp = True) and valid_action_space(action):
                return True
        return False 


    def evaluate_king(self, action, own, hypoth_opp = False):
        """ evaluate if proposed king move is possible"""
        l,n,L,N = action
        if hypoth_opp:
            if (np.abs(L-l) <= 1 and np.abs(N-n) <= 1):
                return True
        else:
            if (np.abs(L-l) <= 1 and np.abs(N-n) <= 1) and not self.checked_position(L,N, own):
                return True
        return False
        

    def evaluate_horse(self, action, hypoth_opp = False):
        """ evaluate if proposed horse move is possible"""
        l,n,L,N = action
        
        if ((np.abs(L-l) == 2 and np.abs(N-n) == 1) or (np.abs(L-l) == 1 and np.abs(N-n) == 2)):
            if not hypoth_opp:
                if self.check_proof_action(action)[0]:
                    return True
            else:
                return True
        return False
    
    
    def evaluate_pawn(self, action, own, hypoth_opp = False):
        """ evaluate if proposed pawn move is possible"""
        sign = 2*int(own)-1
        l,n,L,N = action
        # normal adavencement
        condition1 = ( l == L and N == n+sign and self.chessboard[N,L] == 0)
        # diagonal attack
        condition2 = ( np.abs(L - l)==1 and N == n+sign and np.sign(self.chessboard[N,L]) == -sign)
        
        # double first move
        if own:
            condition3 = ( l == L and n == 1 and N == n+2 and self.chessboard[2,L] == 0 and \
                          self.chessboard[3,L] == 0 and (self.check_board( L-1, 3) >= 0 and self.check_board(L+1,3) >= 0) )
        else:
            condition3 = ( l == L and n == 6 and N == 4 and self.chessboard[5,L] == 0 and \
                          self.chessboard[4,L] == 0 and (self.check_board(L-1,4) <= 0 and self.check_board(L+1,4) <= 0) )
                      
        if condition1 or condition2 or condition3:
            if not hypoth_opp:
                if self.check_proof_action(action)[0]:
                    return True
            else:
                return True
        return False
    
    
    def evaluate_bishop(self, action, hypoth_opp = False):
        """ evaluate if proposed bishop move is possible"""
        l,n,L,N = action
        
        if np.abs(L-l) == np.abs(N-n):
            l_sign = np.sign(L-l)
            n_sign = np.sign(N-n)
        else:
            return False
            
        if all([self.check_board(l+l_sign*i, n+n_sign*i)== 0 for i in range(1,np.abs(L-l))  ] ):
            if not hypoth_opp:
                if self.check_proof_action(action)[0]:
                    return True
            else:
                return True
        return False
                        
    
    def evaluate_tower(self, action, hypoth_opp = False):
        """ evaluate if proposed tower move is possible"""
        l,n,L,N = action
        valid = False
        if L == l  or N  == n:
            if L != l:
                l_sign = np.sign(L-l)
                if all([self.check_board(l+l_sign*i, n) == 0 for i in range(1,np.abs(L-l))  ] ):
                    valid = True
            elif N != n:
                n_sign = np.sign(N-n)
                if all([self.check_board(l, n+n_sign*i) == 0 for i in range(1,np.abs(N-n))  ] ):
                    valid = True
        if valid:
            if not hypoth_opp:
                if self.check_proof_action(action):
                    return True
            else:
                return True
        return False
            
        
def margin_check(in_vector, l = None, n = None):
    """ check if board margin are respected for a vector of board positions (returns only valid ones)"""
    if l is not None:
        equals = np.equal( in_vector ,[l,n]).all(1)
        if any(equals):
            in_vector = in_vector[np.where(np.bitwise_not(equals))]

    return in_vector[np.bitwise_and(in_vector[:,0]<=7, in_vector[:,0]>=0)]


def valid_action_space(action):
    """ checks if action is on board"""
    return all([ 0 <= a <=7 for a in  action ])


def sym_rep(symbol):
    """ render symbols """
    if symbol == 0:
        out = ' '
    elif symbol == 1:
        out = 'P'
    elif symbol == -1:
        out = 'p'
    elif symbol == 5:
        out = 'R'
    elif symbol == -5:
        out = 'r'
    elif symbol == 4:
        out = 'H'
    elif symbol == -4:
        out = 'h'
    elif symbol == 3:
        out = 'B'
    elif symbol == -3:
        out = 'b'
    elif symbol == 9:
        out = 'Q'
    elif symbol == -9:
        out = 'q'
    elif symbol == 10:
        out = 'K'
    elif symbol == -10:
        out = 'k'
    else:
        raise('symbol error')
    
    return out


def possible_moves(piece_idx, l, n, own = True):
    """ returns possible move for a piece, given its position, regardless of pieces on board"""
    if not np.isscalar(l):
        l = l[0]
        n = n[0]
    
    moves = np.zeros((8,8), dtype = bool )
    sign = 2*int(own)-1
    
    # if pawn
    if piece_idx == 1:
        out = np.array([ [l-1,n+sign], [l,n+sign], [l+1,n+sign], [l,n+2*sign] ])
        return margin_check(out, l,n)
    
    # if bishop
    elif piece_idx == 3:
        for i in range(8):
            for j in range(8):
                if np.abs(i-n) == np.abs(l-j):
                    moves[i, j] = True 
        idx = np.where(moves)
        out = np.concatenate((idx[1][:,np.newaxis], idx[0][:,np.newaxis]), axis = 1)
        return margin_check(out, l,n)
    
    # if tower
    elif piece_idx == 5:
        moves[n, :] = True
        moves[:, l] = True
        idx = np.where(moves)
        out =  np.concatenate((idx[1][:,np.newaxis], idx[0][:,np.newaxis]), axis = 1)
        return margin_check(out, l,n)
    
    # if horse
    elif piece_idx == 4:
        for i in range(8):
            for j in range(8):
                moves[ ((np.abs(j-l) == 2 and np.abs(i-n) == 1) or (np.abs(j-l) == 1 and np.abs(i-n) == 2)) ] = True
        idx = np.where(moves)
        out =  np.concatenate((idx[1][:,np.newaxis], idx[0][:,np.newaxis]), axis = 1)
        return margin_check(out, l,n)
    
    # if queen
    elif piece_idx == 9:
        for i in range(8):
            for j in range(8):
                if np.abs(i-n) == np.abs(l-j):
                    moves[i, j] = True 
        moves[n, :] = True
        moves[:, l] = True
        idx = np.where(moves)
        out = np.concatenate((idx[1][:,np.newaxis], idx[0][:,np.newaxis]), axis = 1)
        return margin_check(out, l,n)
    
    # if king
    elif piece_idx == 10:
        out = np.array([ [l-1,n+1], [l,n+1], [l+1,n+1], [l+1,n], [l+1,n-1], [l,n-1], [l-1,n-1], [l-1,n] ])
        return margin_check(out, l,n)
    
    else:
        raise('l,n couple not a piece!')


#%%
if __name__ == "__main__":

    """
    # compiling time debug
    import cProfile
    import pstats
    import io
    pr = cProfile.Profile()
    pr.enable()
    """

    game = Chess(print_out = False)
    game.reset()
    done = False
    
    while not done:
        
        action = game.random_move(own = True)
        state,reward,done,info = game.step(action)
        #print(f'reward = {reward}')
        #print(f'state = {state}')
        
    if len(info) == 0:
        print('wrong move selected!')
    else:
        print(f'the winner is: {info["winner"]}')
        
    """
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()
    with open('duration_Chess.txt', 'w+') as f:
        f.write(s.getvalue())
    """
