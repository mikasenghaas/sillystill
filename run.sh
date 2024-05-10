#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 10G
#SBATCH --time 30:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552
#SBATCH --output=logs/slurm/slurm-%j.out

# Load modules
module load gcc python 

# Activate venv
source ~/venvs/sillystill/bin/activate

# Run Python script
python src/train.py logger=wandb model/loss_fn=cobi logger.name=with_cobi

# Deactive venv
deactivate
