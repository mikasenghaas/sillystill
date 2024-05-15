#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --time 3:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552
#SBATCH --output=logs/slurm/slurm-%j.out

# Load modules
module load gcc python 

# Activate venv
source ~/venvs/sillystill/bin/activate

# Install requirements
pip install -r requirements.txt

# Other available experiments
# python src/train.py -m experiment=translation-net
# python src/train.py -m experiment=translation-patch
# python src/train.py -m experiment=translation-loss
# python src/train.py -m experiment=translation-augment

# Deactive venv
deactivate
