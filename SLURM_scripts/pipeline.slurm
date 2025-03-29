#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=dlp/results/results_%j.txt      
#SBATCH --error=dlp/results/error_%j.txt         
#SBATCH --ntasks=1                         
#SBATCH --cpus-per-task=4                 
#SBATCH --mem=4G                           
#SBATCH --time=01:00:00                    
#SBATCH --partition=short                  
#SBATCH --mail-type=END,FAIL              
#SBATCH --mail-user=j.toot@student.rug.nl 

source ~/envs/DLP/bin/activate