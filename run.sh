#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 50G
#SBATCH --time 1:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs413
#SBATCH --qos cs413
#SBATCH --output=logs/slurm/slurm-%j.out

# Load modules
module load gcc python 

# Activate venv
source ~/venvs/sillystill/bin/activate

# Run Python script
python src/train.py -m experiment=paired

# Deactive venv
deactivate