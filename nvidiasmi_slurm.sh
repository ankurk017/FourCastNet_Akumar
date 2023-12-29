#!/bin/bash
#SBATCH --mail-user=akumar@nsstc.uah.edu  
#SBATCH -J FCN
#SBATCH --gres=gpu:a100
#SBATCH --ntasks 2
#SBATCH -t 20-00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=END,FAIL
#SBATCH -o slurm-%j.out # STDOUT
#SBATCH -e slurm-%j.err # STDERR

########## Add all commands below here


nvidia-smi
