#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH --time 8:00:00
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

# Method 1: Translation Networks
# python src/train.py -m experiment=unet-mse
# python src/train.py -m experiment=unet-cobi
# python src/train.py -m experiment=unet-mse-vgg
# python src/train.py -m experiment=unet-mse-vgg-augment

# Method 2: Autoencoder
# python src/train.py -m experiment=auto-unet-mse-vgg

# Deactive venv
deactivate
