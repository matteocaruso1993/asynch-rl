#!/bin/bash

PATH="/home/fcairoli/miniconda3/bin:/home/fcairoli/miniconda3/condabin:$PATH"
source .bashrc

eval "$(conda shell.bash hook)"
conda activate rl_env

#echo "$PATH"

for ARGUMENT in "$@"
do
    echo $ARGUMENT
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
            DIFF)              DIFF=${VALUE} ;;
            VERS)             VERS=${VALUE} ;;
            ITER)             ITER=${VALUE} ;;
            *)
    esac
done

#RL='AC'

#conda activate venv
#cd git/asynch-rl/

#python examples/Trainer_robot.py -p True -a 25 -v $VERS -i 2000 -l $ITER -d $DIFF
