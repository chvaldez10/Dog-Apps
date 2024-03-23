#!/bin/bash

####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

source ~/software/init-conda
<<<<<<< Updated upstream
# conda activate chris_env
conda activate enel645
python /home/reoredge.santillan/645_assignment_2/enel645_a2_efficientnet_b4.py
# python "/home/christian.valdez/ENSF-611-ENEL-645/ENEL 645/A2/notebooks/a2.py"
=======
conda activate pytorch
python "/home/alton.wong/645_assignment_2/alton_a2.py"
>>>>>>> Stashed changes
conda deactivate