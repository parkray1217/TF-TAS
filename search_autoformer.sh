#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=1 --use_env search_autoformer.py --data-path './dataset/cifar100' --data-set 'CIFAR100'  --gp \
 --change_qk --relative_position  --dist-eval --cfg './experiments/search_space/space-S.yaml' --output_dir './OUTPUT_CIFAR_100/search1'


