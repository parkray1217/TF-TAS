#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=mcs.gpu.q
#SBATCH --error=./result-file/configcheck_%j.err
#SBATCH --output=./result-file/configcheck_%j.out
#SBATCH --time=24:00:00
#SBATCH --constraint=v100
#SBATCH --gres=gpu:2

CONFIG_YAML="./experiments/cifar10-config/config${CONFIG_NUM}.yaml"
OUTPUT_DIR="./cifar10output/config${CONFIG_NUM}"
$COMMAND '$CONFIG_YAML' --output_dir '$OUTPUT_DIR'
