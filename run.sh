#!/bin/bash

#SBATCH --job-name=bash
#SBATCH --partition=LocalQ          # Ensure this partition exists
#SBATCH --nodes=1
#SBATCH --gres=gpu:2                # Request one GPU per job
#SBATCH --mem=70GB                  # Memory request should have '='
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

#SBATCH --error=/home/chizhang/mycode/DiT/logs/errors.err  # Use 'error' instead of 'err'
#SBATCH --output=/home/chizhang/mycode/DiT/logs/output.out  # Use 'output' instead of 'out'

# Full path to the directory containing your Python files
proj_dir="/home/chizhang/mycode/DiT"

cd ${proj_dir}

conda_env="/home/chizhang/myprefix/anaconda3/envs/dit"

# Activate the Conda environment
# conda init
# conda init bash
source /home/chizhang/myprefix/anaconda3/etc/profile.d/conda.sh
conda activate $conda_env
export CUDA_HOME=$conda_env
export CUDA_VISIBLE_DEVICES=2,3

echo "============================================================================================"
echo "Starting training taks"
echo "============================================================================================"

torchrun --nnodes=1 --nproc_per_node=2 ./scripts/train.py --model DiT-S/8 --data-path ./assets/tiny-imagenet-200/train --epochs 10  --log-every 10 --ckpt-every 10

echo "============================================================================================"
echo "Task completed."
echo "============================================================================================"

# Deactivate the Conda environment
# conda end
conda deactivate

wait # Wait for all experiment types to finish
echo "All experiments completed"