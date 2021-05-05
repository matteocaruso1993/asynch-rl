#!/bin/bash

for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            DIFF)              DIFF=${VALUE} ;;
            VERS)    	      VERS=${VALUE} ;;     
            ITER)    	      ITER=${VALUE} ;;     
            SIM)    	      SIM=${VALUE} ;;     
            SAVE)    	      SAVE=${VALUE} ;;     
            RL)    	      RL=${VALUE} ;; 
            *)   
    esac    


done

echo "DIFF = $DIFF"
echo "VERS = $VERS"
echo "ITER = $ITER"
echo "SIM = $SIM"
echo "SAVE = $SAVE"
echo "RL = $RL"


#if [ $RL == 'AC' ]
#then
#    server_string="167:GitHub_repos"
#elif [ $RL == 'DQL' ]
#then
#    server_string="156:GitHub_Repositories"
#fi

server_string="167:GitHub_repos"
echo "server string = $server_string"

source ~/environments/venv_PedEnv/bin/activate

folder="ConvModel$VERS"
iteration="_$ITER"

#echo $folder
#echo $iteration

#mkdir /home/rodpod21/GitHubRepositories/asynch-rl/Data/RobotEnv/${folder}
 
#rm /home/rodpod21/GitHubRepositories/asynch-rl/Data/RobotEnv/${folder}/TrainingLog.pkl

sshpass -p 'abcABC11!?' ssh eregolin@172.30.121.167 "ls "
nohup sshpass -p 'abcABC11!?' ssh eregolin@172.30.121.167 "nohup bash -sl < perform_training.sh VERS='$VERS' ITER='$ITER'" > /home/rodpod21/GitHubRepositories/asynch-rl/Data/RobotEnv/${folder}/nohup.log 2>&1 

#nohup command > output.log 2>&1

# # VERS='$VERS' ITER='$ITER'

#python /home/rodpod21/GitHubRepositories/asynch-rl/examples/Tester_robot.py -v $VERS -i $ITER -sim $SIM -d $DIFF -s $SAVE
