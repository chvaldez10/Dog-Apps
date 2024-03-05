#!/bin/bash

####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

source ~/software/init-conda


conda activate enel645

# jupyter nbconvert --execute --to notebook --inplace /home/reoredge.santillan/645_assignment_2/enel645_a2.ipynb --best_model_path /home/reoredge.santillan/645_assignment_2/garbage_net.pth --images_path /work/TALC/enel645_2024/CVPR_2024_dataset/ --transfer_learning True
jupyter nbconvert --to script /home/reoredge.santillan/645_assignment_2/enel645_a2.ipynb
python /home/reoredge.santillan/645_assignment_2/enel645_a2.py --best_model_path /home/reoredge.santillan/645_assignment_2/garbage_net.pth --images_path /work/TALC/enel645_2024/CVPR_2024_dataset/ --transfer_learning True

conda deactivate
