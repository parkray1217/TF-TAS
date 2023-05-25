#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=1 --use_env search_autoformer.py --data-path './dataset/cifar10' --data-set 'Indian' --indicator-name 'dss' --gp \
 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-hyper.yaml' --output_dir './OUTPUT-indian/indiansearch2'