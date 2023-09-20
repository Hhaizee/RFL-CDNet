#!/bin/bash

#SBATCH -A jhliu4          
#SBATCH -J  pp_work
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH -o out_scream_train.out
#SBATCH -t 5-1:30:00

source activate zyh_cd
##source activate liu_changedetection

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29509 train.py