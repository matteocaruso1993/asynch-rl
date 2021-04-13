#!/bin/bash

source ~/environments/venv_PedEnv/bin/activate

folder="ConvModel$1"
iteration="_$2"

#echo $folder
#echo $iteration

mkdir /home/rodpod21/GitHubRepositories/asynch-rl/Data/RobotEnv/${folder}
 
sshpass -p 'abcABC11!?' scp eregolin@172.30.121.167:GitHub_repos/asynch-rl/Data/RobotEnv/${folder}/\{'TrainingLog.pkl','train_params.txt','*'$iteration'*','val_history.npy','PG_training.npy'\} /home/rodpod21/GitHubRepositories/asynch-rl/Data/RobotEnv/${folder}/

python /home/rodpod21/GitHubRepositories/asynch-rl/examples/Tester_robot.py -v $1 -i $2 -sim True
