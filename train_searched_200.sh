#!/bin/bash

nproc_per_node=1
data_path='./dataset/cifar10'
data_set='CIFAR10'
gp=''
change_qk=''
relative_position=''
mode='retrain'
model_type='AUTOFORMER'
dist_eval=''
cfg_base='./experiments/cifar10-config/config'
output_base='./cifar10output/config1-1-nocpu-'

for i in {1..1}; do
    sbatch --job-name=train-$i \
        --output=./result-file/configcheck_1_cpu.out \
        --error=./result-file/configcheck_1_cpu.err\
        --nodes=1 \
        --ntasks-per-node=1 \
        --gres=gpu:8 \
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
