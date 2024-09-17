#!/bin/bash

# Add the Git LFS repository using the script provided by packagecloud
echo "Downloading Git LFS binaries"
wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz
tar -xzvf git-lfs-linux-amd64-v3.5.1.tar.gz
mkdir -p ~/bin
mv git-lfs-3.5.1/git-lfs ~/bin/
rm -rf git-lfs-linux-amd64-v3.5.1.tar.gz
rm -rf git-lfs-3.5.1rm -rf git-lfs-3.5.1
export PATH=$HOME/bin:$PATH
source ~/.bashrc

# Set up Git LFS
echo "Setting up Git LFS..."
git lfs install

# Create venv
echo "Creating virtual environment..."
virtualenv --system-site-packages ~/venvs/sillystill

# Activate venv
source ~/venvs/sillystill/bin/activate
echo "Installing packages from requirements.txt"
pip install --upgrade pip
pip install -r requirements.txt

# Loading .env
echo "Loading .env"
if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi

# Setup W&B API key
echo "Setting up W&B API key"
python -m wandb login $WANDB_KEY