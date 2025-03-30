#!/bin/bash
#SBATCH --job-name=symb_regression
#SBATCH --output=/home2/s3306801/github/DeepLearningPH/results/results_pysr%j.txt      
#SBATCH --error=/home2/s3306801/github/DeepLearningPH/results/error_pysr%j.txt
#SBATCH --chdir=/home2/s3306801/github/DeepLearningPH         
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=regular                                   
#SBATCH --mem=4G                           
#SBATCH --time=01:00:00                                 
#SBATCH --mail-type=END,FAIL              
#SBATCH --mail-user=j.toot@student.rug.nl 

module load Python/3.9.5-GCCcore-10.3.0

source ~/envs/DLP/bin/activate

echo "Running on $(hostname)"
echo "Python path: $(which python)"

export PYTHONPATH=/home2/s3306801/github/DeepLearningPH:$PYTHONPATH

python /home2/s3306801/github/DeepLearningPH/SLURM_scripts/pysr_slurm.py
