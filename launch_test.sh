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
            *)   
    esac    
done

#echo "DIFF = $DIFF"
#echo "VERS = $VERS"
#echo "ITER = $ITER"
#echo "SIM = $SIM"
#echo "SAVE = $SAVE"

### enrico local
source ~/environments/venv_PedEnv/bin/activate
local_path_string="/home/rodpod21/GitHubRepositories"
### ros
#source ~/CrowdNav/bin/activate
#local_path_string="~/crowd"

python ${local_path_string}/asynch-rl/examples/Tester_robot.py -v $VERS -i $ITER -sim $SIM -d $DIFF -s $SAVE
