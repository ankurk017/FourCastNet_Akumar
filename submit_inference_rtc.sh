#!/bin/bash
#SBATCH --mail-user=akumar@nsstc.uah.edu  
#SBATCH -J FCN
#SBATCH -p shared --gres=gpu:rtx5000:1
#SBATCH --ntasks 2
#SBATCH -t 20-00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=END,FAIL
#SBATCH -o slurm-%j.out # STDOUT
#SBATCH -e slurm-%j.err # STDERR

########## Add all commands below here

export PATH="/rhome/akumar/anaconda3/envs/dlwp_new/bin:$PATH"

python inference/inference.py --config=afno_backbone  --run_num=0 --weights /nas/rstor/akumar/USA/PhD/FourCastNet/Training_38vars_testing/FourcastNet/exp/afno_backbone_finetune/0/training_checkpoints/best_ckpt.tar --override_dir /nas/rstor/akumar/USA/PhD/FourCastNet/Training_38vars_testing/inference_output  --vis 

echo '**********Finished!**********'

