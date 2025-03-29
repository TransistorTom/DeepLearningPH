#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=/home2/github/DeepLearningPH/results/results_%j.txt      
#SBATCH --error=/home2/github/DeepLearningPH/results/error_%j.txt         
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpushort
#SBATCH --mem=4G                           
#SBATCH --time=01:00:00                                 
#SBATCH --mail-type=END,FAIL              
#SBATCH --mail-user=j.toot@student.rug.nl 

echo "Job started on $(hostname) at $(date)"
echo "Current working directory: $(pwd)"
env > /home2/github/DeepLearningPH/results/env_$SLURM_JOB_ID.txt

module load Python/3.9.5-GCCcore-10.3.0
module load CUDA

source ~/envs/DLP/bin/activate || { echo "Failed to activate env"; exit 1; }

ls -l ~/DeepLearningPH/pipeline_runner_slurm.py || exit 1

nvidia-smi || echo "nvidia-smi failed"

time -v python ~/DeepLearningPH/pipeline_runner_slurm.py
