from fabric import Connection
import pandas as pd
import numpy as np
import time
import os
import subprocess
import signal

import server_config

asynch_rl_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

remote_relaunch = True
net_version = str(700)
difficulty = str(1)

print('preparing to connect')
host = server_config.username+"@"+server_config.ip_address
password = server_config.password
c = Connection(host=host,connect_kwargs={"password":password})
print('connection completed')

result=c.run("cat /home/" + server_config.username+"/git/asynch-rl/Data/RobotEnv/ConvModel"+net_version+"/train_log.txt")

print("data extracted")

df = pd.DataFrame.from_records([ i.split(';') for i in result.stdout.split('\n')][:-1], columns = ['iteration', 'date'])

df.loc[-1] = df.loc[0]  # adding a row
df.index = df.index + 1  # shifting index
df.sort_index(inplace=True) 

df['date']= pd.to_datetime(df['date'])

df['duration']= 0.0
df['duration'][1:]= (df['date'][1:].values-df['date'][:-1].values)/ np.timedelta64(1, 's')


last_iteration = df.iloc[-1]['iteration']
df = df[2:]

print(f'{df[-20:]}')
print(f'last iteration: {last_iteration}')



#video_condition = False and not int(last_iteration) % 20

video_condition = True
    
makedir_string = "mkdir "+asynch_rl_path+"/asynch-rl/Data/RobotEnv/ConvModel"+net_version+"/"
 
copy_string = "sshpass -p "+server_config.password+" scp "+server_config.username+"@"+server_config.ip_address+":git/asynch-rl/Data/RobotEnv/ConvModel"\
    +net_version+"/\{'TrainingLog.pkl','train_params.txt','*'$iteration'*','val_history.npy','PG_training.npy'\} "+asynch_rl_path+"/asynch-rl/Data/RobotEnv/ConvModel"+net_version+"/"

print(copy_string)

os.system(makedir_string)
os.system(copy_string)

os.system("bash "+ asynch_rl_path +"/asynch-rl/launch_test1.sh VERS='"+ net_version +"' ITER='"+last_iteration+"' DIFF='"+str(difficulty) +"' SIM='"+ str(video_condition) +"' SAVE='True' RL='AC'")

time.sleep(5)


os.system("cp "+ asynch_rl_path +"/asynch-rl/Data/RobotEnv/ConvModel"+net_version+"/video/*.png  ~/Dropbox/CrowdNavigationTraining")

"""
if video_condition:
    time.sleep(200)
    os.system("cp "+ asynch_rl_path +"/asynch-rl/Data/RobotEnv/ConvModel"+net_version+"/video/*"+str(last_iteration)+"*  ~/Dropbox/CrowdNavigationTraining")
"""


current_duration = round(time.time() - df.iloc[-1]['date'].timestamp())

print(f'current duration: {current_duration}s')


"""
if current_duration > 300: #3*df[df['duration']<300]['duration'].mean():
    os.system("rm ~/Dropbox/CrowdNavigationTraining/SIMULATION_STALLED_*")
    os.system("touch ~/Dropbox/CrowdNavigationTraining/SIMULATION_STALLED_"+ last_iteration)
    
    if remote_relaunch:
        try:
            c.run("kill -9 -1 -u "+server_config.username)
        except Exception:
            print('pseudo error after kill')
        
        relaunch_string =  "nohup sshpass -p '"+password+"' ssh "+host+" \"nohup bash -sl < perform_training.sh DIFF='"+difficulty+"' VERS='"+net_version+"' ITER='"+last_iteration+"'\" > "+asynch_rl_path+\
            "/asynch-rl/Data/RobotEnv/ConvModel" +net_version+"/nohup.log 2>&1 "

        os.system(relaunch_string)               

        #relaunch_command = "nohup bash "+ asynch_rl_path +"/asynch-rl/launch_training.sh 'ITER'=" + last_iteration + " 'VERS'=" + net_version + " &" 
        #os.system(relaunch_command)
        time.sleep(250)
        print('########### relaunch completed ###########')
    
else:
    print('########### iteration went well. simulation advancing...###########')
    os.system("rm ~/Dropbox/CrowdNavigationTraining/last_successful*")
    os.system("touch ~/Dropbox/CrowdNavigationTraining/last_successful_"+ last_iteration +" &")
"""

    
    
