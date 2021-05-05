from fabric import Connection
import pandas as pd
import numpy as np
import time
import os
import subprocess
import signal

remote_relaunch = True
net_version = str(330)

print('preparing to connect')
c = Connection(host="eregolin@172.30.121.167",connect_kwargs={"password":"abcABC11!?"})
print('connection completed')

result=c.run("cat /home/eregolin/GitHub_repos/asynch-rl/Data/RobotEnv/ConvModel"+net_version+"/train_log.txt")

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



video_condition = False and not int(last_iteration) % 20

os.system("bash ~/GitHubRepositories/asynch-rl/copy_sim_data.sh VERS='"+ net_version +"' ITER='"+last_iteration+"' DIFF='3' SIM='"+ str(video_condition) +"' SAVE='True' RL='AC'")

os.system("cp /home/rodpod21/GitHubRepositories/asynch-rl/Data/RobotEnv/ConvModel"+net_version+"/video/*.png  /home/rodpod21/Dropbox/CrowdNavigationTraining")

if video_condition:
    time.sleep(200)
    os.system("cp /home/rodpod21/GitHubRepositories/asynch-rl/Data/RobotEnv/ConvModel"+net_version+"/video/*"+str(last_iteration)+"*  /home/rodpod21/Dropbox/CrowdNavigationTraining")

current_duration = round(time.time() - df.iloc[-1]['date'].timestamp())

print(f'current duration: {current_duration}s')


if current_duration > 180: #3*df[df['duration']<300]['duration'].mean():
    os.system("rm /home/rodpod21/Dropbox/CrowdNavigationTraining/SIMULATION_STALLED_*")
    os.system("touch /home/rodpod21/Dropbox/CrowdNavigationTraining/SIMULATION_STALLED_"+ last_iteration)
    
    if remote_relaunch:
        try:
            c.run("kill -9 -1 -u eregolin")
        except Exception:
            print('pseudo error after kill')
        
        relaunch_command = "nohup bash /home/rodpod21/GitHubRepositories/asynch-rl/launch_training.sh 'ITER'=" + last_iteration + " 'VERS'=" + net_version + " &" 
        #relaunch_command = [ "nohup","bash", "/home/rodpod21/GitHubRepositories/asynch-rl/launch_training.sh", "'ITER'="+last_iteration, "'VERS'="+net_version , "&" ]
        os.system(relaunch_command)
        time.sleep(150)
        print('########### relaunch completed ###########')
        
        
    
else:
    print('########### iteration went well. simulation advancing...###########')
    os.system("rm /home/rodpod21/Dropbox/CrowdNavigationTraining/last_successful*")
    os.system("touch /home/rodpod21/Dropbox/CrowdNavigationTraining/last_successful_"+ last_iteration +" &")

    #p_list = subprocess.run(["touch", "/home/rodpod21/Dropbox/CrowdNavigationTraining/last_successful_"+ last_iteration])
    #os.killpg(p_list.pid, signal.SIGTERM)
    
    