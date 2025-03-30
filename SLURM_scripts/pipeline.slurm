#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=/home2/s3306801/github/DeepLearningPH/results/results_%j.txt      
#SBATCH --error=/home2/s3306801/github/DeepLearningPH/results/error_%j.txt
#SBATCH --chdir=/home2/s3306801/github/DeepLearningPH         
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpushort                                      
#SBATCH --mem=4G                           
#SBATCH --time=01:00:00                                 
#SBATCH --mail-type=END,FAIL              
#SBATCH --mail-user=j.toot@student.rug.nl 

module load Python/3.9.5-GCCcore-10.3.0
module load CUDA

source ~/envs/DLP/bin/activate

echo "Running on $(hostname)"
echo "Python path: $(which python)"

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv -l 5 > gpu_usage.log &
trap "kill $!" EXIT

export PYTHONPATH=/home2/s3306801/github/DeepLearningPH:$PYTHONPATH

python /home2/s3306801/github/DeepLearningPH/SLURM_scripts/pipeline_slurm.py
