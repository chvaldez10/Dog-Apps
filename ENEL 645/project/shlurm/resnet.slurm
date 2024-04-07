#!/bin/bash

####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

source /home/christian.valdez/software/miniconda3/etc/profile.d/conda.sh
conda activate calgary_di
python "/home/christian.valdez/ENSF-611-ENEL-645/ENEL 645/project/resnet_50.py" --train --test
conda deactivate