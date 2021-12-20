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
from cProfile import label
import matplotlib.pyplot as plt
import sys
import zipfile


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
    storage_path = os.path.join( os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) ,"Data" , \
                            env_type, model_type+str(net_version) )
    
    with open(os.path.join(storage_path,'train_params.txt'), 'r+') as f:
        Lines = f.readlines() 
        
    for line in Lines: 
        for var in overwrite_params:
            if var in line: #[: line.find(':')]:
                if 'bool' in line:
                    if 'False' in line:
                        my_dict.__setitem__(var , False )
                    else:
                        my_dict.__setitem__(var , True )
                else:
                    my_dict.__setitem__(var , eval(line[line.find(':')+1 : line.find('\n')])) 
                
    return my_dict


def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was



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




def loadPlotPieTester(traj_stat_filename='traj_stats.txt', line_no=None):
    print('test')
    lines = [[v for v in line.split()] for line in open(traj_stat_filename)]
    tot_runs = 0
    n_ped = 0
    n_obs = 0
    n_success = 0
    n_timeout = 0

    if line_no is None:
        for line in lines:
            tot_runs += len(line)
            n_ped += line.count('pedestrian')
            n_obs += line.count('obstacle')
            n_timeout += line.count('timeout')
            n_success += line.count('success')

    else:
        line = lines[line_no]
        n_ped += line.count('pedestrian')
        n_obs += line.count('obstacle')
        n_timeout += line.count('timeout')
        n_success += line.count('success')
        

    print('Total number of simulations:\t%d'%tot_runs)
    vals = [n_success,n_ped,n_obs,n_timeout]
    vals_principal = [n_success,n_ped + n_obs + n_timeout]
    vals_secondary = [n_ped,n_obs,n_timeout]
    print(vals)
    labels_principal_pie = ['success','failure']
    colors_principal_pie = ['#00ff00','#ff0000']

    labels_secondary_pie = ['pedestrian', 'obstacle','timeout']
    colors_secondary_pie = ['#ff0000','#ff9933','#0000ff']
    center = (2,2)
    explode_principal = (0,0)
    explode_secondary = (0,0,0)
    wedgeprops={"edgecolor":"k",'linewidth': 1, 'linestyle': 'solid', 'antialiased': True}
    fig,ax = plt.subplots()
    ax.pie(vals_principal,colors=colors_principal_pie,explode = explode_principal, labels=labels_principal_pie,autopct='%1.1f%%',shadow=False,startangle=90,wedgeprops=wedgeprops)
    ax.pie(vals_secondary,radius=0.50,center=center,colors=colors_secondary_pie,explode = explode_secondary, labels=labels_secondary_pie,autopct='%1.1f%%',shadow=False,startangle=90,wedgeprops=wedgeprops)

    ax.axis('equal')
    plt.tight_layout()
    plt.show()




def zipModel(env,model_number):
    parent = os.path.dirname(__file__)
    pack_folder = os.path.abspath(os.path.join(parent,'..','..'))
    model_folder = os.path.join(pack_folder,'Data',env,model_number)
    print(model_folder)
    relroot = os.path.abspath(os.path.join(model_folder, os.pardir))
    tmp = zipfile.ZipFile('tmp.zip','w',zipfile.ZIP_DEFLATED)
    for dirname, subdirs, files in os.walk(model_folder):
        tmp.write(dirname, os.path.relpath(dirname,relroot))
        for file in files:
            filename = os.path.join(dirname, file)
            arch_name = os.path.join(os.path.relpath(dirname,relroot), file)
            tmp.write(filename,arch_name)

    tmp.close()
