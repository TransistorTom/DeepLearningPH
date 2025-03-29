#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=dlp/results/results_%j.txt       # Save output in logs dir
#SBATCH --error=dlp/results/error_%j.txt         # Save error output
#SBATCH --ntasks=1                         # Number of tasks (1 = serial job)
#SBATCH --cpus-per-task=4                 # Adjust for your script
#SBATCH --mem=4G                           # Memory per node
#SBATCH --time=01:00:00                    # Time limit (HH:MM:SS)
#SBATCH --partition=short                  # Partition name (e.g., short, long)
#SBATCH --mail-type=END,FAIL              # Get email when job ends or fails
#SBATCH --mail-user=j.toot@student.rug.nl # Your email