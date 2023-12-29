#!/bin/bash
### SLURM batch script

### Email address
#SBATCH --mail-user=ankur.kumar@nsstc.uah.edu

### Job name
#SBATCH -J FCN

### Partition (queue), select shared queue only for GPUs
### Optionally specify a GPU type: --gres=gpu:rtx5000:1 or --gres-cpu:a100:1
#SBATCH -p shared --gres=gpu:rtx5000:1

### TOTAL processors (number of tasks)
#SBATCH --ntasks 4

### total run time estimate (D-HH:MM)
#SBATCH -t 0-01:00

### allocated memory for EACH processor (GB))
### NOTE: TOTAL MEMORY ALLOCATED = mem-per-cpu X ntasks
### Be careful about making this number large, especially when using a large number of processors (ntasks)
#SBATCH --mem-per-cpu=1G

### Mail to user on an event
### common options are FAIL, BEGIN, END, REQUEUE
#SBATCH --mail-type=END,FAIL

### Ouput files
#SBATCH -o slurm-%j.out # STDOUT
#SBATCH -e slurm-%j.err # STDERR


echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo

### Add your module statments here
module load cuda
$CUDA_PATH/samples/bin/x86_64/linux/release/deviceQuery



export PATH="/rhome/akumar/anaconda3/envs/dlwp/bin:$PATH"

#python inference/inference.py --config=afno_backbone  --run_num=0 --weights ../npy/backbone.ckpt  --override_dir  ../output_v1/ERA5_copernicus --vis 

config_file=./config/AFNO.yaml
config='afno_backbone_finetune'
run_num='0'

python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
echo '**********Finished!**********'

