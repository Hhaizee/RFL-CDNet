#!/bin/bash

#SBATCH -A jhliu4          
#SBATCH -J  pp_work
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH -o out_scream_train_v3output.out
#SBATCH -t 5-1:30:00
#SBATCH --exclude=g0015


source activate liu_changedetection
echo "Starting job"
python visualizationo1.py
echo "Job finished"