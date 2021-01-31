# The MIT License (MIT)
# Copyright (c) 2016 Vladimir Ignatev
#
# Permission is hereby granted, free of charge, to any person obtaining 
# a copy of this software and associated documentation files (the "Software"), 
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software 
# is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
import sys


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
"""

##########################################################################   
import sys


def progress(count, total, status='', filled_symbol = '='):
    red='\033[01;31m'
    gre='\033[02;32m'
    yel='\033[00;33m'
    blu='\033[01;34m'
    res='\033[00m'
    
    
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))
    
    percentage = round(100.0 * count / float(total), 1)
    bar = filled_symbol * filled_len + '-' * (bar_len - filled_len)
    
    if percentage <= 50:
        col = red
    elif percentage > 50 and percentage <= 75:
        col = yel
    elif percentage > 75 and percentage <= 90:
        col = gre
    else:
        col = res
    
    sys.stdout.write('\r{0}[{1}] {2}%  {3}'.format(col, bar, percentage, status))
    
    
    sys.stdout.flush()
    
##########################################################################   
"""This provides a lineno() function to make it easy to grab the line
number that we're on.

Danny Yoo (dyoo@hkn.eecs.berkeley.edu)
"""

import inspect

def lineno(print_flag = True):
    """Returns the current line number in our program."""
    lineno = inspect.currentframe().f_back.f_lineno
    if print_flag:
        print(f'line number {lineno}')
    return lineno


if __name__ == '__main__':
    lineno()
    print('') 
    print('') 
    lineno()
    

##########################################################################    
import shutil
import os

# clear pycache from tree
def clear_pycache(path):

    for directories, subfolder, files in os.walk(path):
        if os.path.isdir(directories):
            if directories[::-1][:11][::-1] == '__pycache__':
                            shutil.rmtree(directories)
                            
                   
##########################################################################
#save/load training parameters to txt file (only first time train launch)
def store_train_params(rl_env, function_inputs):
    
    with open(os.path.join(rl_env.storage_path,'train_params.txt'), 'w+') as f:
        
        for input_name in function_inputs:
                f.writelines(str(input_name)+': ')
                if isinstance(function_inputs[str(input_name)], str):
                    f.writelines("'" + str(function_inputs[str(input_name)])+"'")
                elif isinstance(function_inputs[str(input_name)], bool):
                    f.writelines("bool('" + str(function_inputs[str(input_name)])+"')")
                else :
                    f.writelines(str(function_inputs[str(input_name)]))
                # to add data you only add String data so you want to type cast variable  
                f.writelines("\n")
    
    
def load_train_params(env_type, model_type, overwrite_params, net_version):
    my_dict = {}
    storage_path = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) ,"Data" , \
                            env_type, model_type+str(net_version) )
    
    with open(os.path.join(storage_path,'train_params.txt'), 'r+') as f:
        Lines = f.readlines() 
        
    for line in Lines: 
        for var in overwrite_params:
            if var in line: #[: line.find(':')]:
                my_dict.__setitem__(var , eval(line[line.find(':')+1 : line.find('\n')])) 
                
    return my_dict


############################################################################
import time
import numpy as np

def check_WhileTrue_timeout(time_in, t_max = 60, printout = False):
    time_exp = np.round(time.time() - time_in,4)
    if time_exp > t_max:
        print(f'time expired: {time_exp}')
        return True
    if printout:
        if np.abs( time_exp - np.round(time_exp)) < 1e-3:
            print(f'time expired: {np.round(time_exp)}')
            time.sleep(2e-3)
    return False
    

def check_saved(filename):
    counter = 0
    while not os.path.isfile(filename):
        time.sleep(1)
        counter += 1
        if not counter % 5:
            print(f'saving {filename}, {counter} seconds')
        if counter >= 60:
            print(f'file not saved: {filename}')
            raise('fatal error!')
