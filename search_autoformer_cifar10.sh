#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=1 --use_env search_autoformer.py --data-path './dataset/cifar10' --data-set 'CIFAR10' --indicator-name 'dss' --gp \
 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-T-1.yaml' --output_dir './OUTPUT-SAMPLE500/dss500search6'