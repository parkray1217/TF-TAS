#!/bin/bash

nproc_per_node=1
data_path='./dataset/cifar10'
data_set='Indian'
gp=''
change_qk=''
relative_position=''
mode='retrain'
model_type='AUTOFORMER'
dist_eval=''
cfg_base='./experiments/indian-config1/configspace-hyper2-'
output_base='./indianoutput-spacehyper/config'

for i in {1..200}; do
    sbatch --job-name=train-$i \
        --output=./result-file-hyper/hyper2/config_%j.out \
        --error=./result-file-hyper/hyper2/config_%j.err\
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=4 \
        --constraint=v100 \
        --gres=gpu:2 \
        --time=24:00:00 \
        --partition=mcs.gpu.q \
        --wrap="python3 -m torch.distributed.launch \
            --nproc_per_node=$nproc_per_node \
            --use_env train.py \
            --data-path=$data_path \
            --data-set=$data_set \
            $gp $change_qk $relative_position \
            --mode $mode \
            --model_type '$model_type' \
            $dist_eval \
            --cfg '$cfg_base$i.yaml' \
            --output_dir '$output_base$i' "
done
