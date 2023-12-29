#!/bin/bash

#SBATCH -A jhliu4          
#SBATCH -J  pp_work
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -o out_scream_count2_lzm.out
#SBATCH -t 5-1:30:00
#SBATCH --exclude=g0015,g0005,g0010


source activate zyh_cd
echo "Starting job"
python count.py
echo "Job finished"