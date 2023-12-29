#!/bin/bash
#SBATCH --mail-user=akumar@nsstc.uah.edu  
#SBATCH -J FCN
#SBATCH --gres=gpu:a101
#SBATCH --ntasks 2
#SBATCH -t 20-00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=END,FAIL
#SBATCH -o slurm-%j.out # STDOUT
#SBATCH -e slurm-%j.err # STDERR

########## Add all commands below here

export PATH="/rhome/akumar/anaconda3/envs/dlwp_new/bin:$PATH"
module load cuda

config_file=./config/AFNO_inference_merra38.yaml

python inference/inference.py --yaml_config=$config_file  --config=afno_backbone  --run_num=0 --weights /nas/rstor/akumar/USA/PhD/FourCastNet/Training_38vars_testing/FourcastNet/exp/afno_backbone_finetune/0/training_checkpoints/best_ckpt.tar --override_dir /rgroup/stela/akumar/FourCastNet_Inference_output/MERRA_38  --vis 

echo '**********Finished!**********'

