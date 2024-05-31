#!/bin/bash

####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

source ~/software/init-conda

conda activate mlenv

python /home/cmychung/enel645_project/enel_645_project.py

conda deactivate
