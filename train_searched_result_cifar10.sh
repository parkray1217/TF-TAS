#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --data-path './dataset/cifar10' --data-set 'CIFAR10' --gp --change_qk --relative_position \
--mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/cifar10-config/config${i}.yaml' --output_dir './cifar10output/config${i}'