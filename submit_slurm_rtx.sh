#!/bin/bash
#SBATCH --mail-user=akumar@nsstc.uah.edu  
#SBATCH -J FCN
#SBATCH -p shared --gres=gpu:rtx5000:1
#SBATCH --ntasks 1
#SBATCH -t 20-00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=END,FAIL
#SBATCH -o slurm-%j.out # STDOUT
#SBATCH -e slurm-%j.err # STDERR

########## Add all commands below here

export PATH="/rhome/akumar/anaconda3/envs/dlwp_new/bin:$PATH"
module load cuda

#python inference/inference.py --config=afno_backbone  --run_num=0 --weights ../npy/backbone.ckpt  --override_dir  ../output_v1/ERA5_copernicus --vis 

config_file=./config/AFNO.yaml
config='afno_backbone_finetune'
run_num='0'

python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
echo '**********Finished!**********'

