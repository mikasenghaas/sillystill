# Sillystill - Recreate the look of Cinestill-800T using Deep Learning

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

![Example of Image captured on Cinestill-800T](imgs/cinestill-800t.jpg)

While halation and chromatic aberrations are sometimes considered as defects, artists often use these optical flaws for aesthetic purposes. Most films contain an anti-halation filter, but the Cinestill 800T does not. This produces a beautiful red halo around light sources, and has a Tungsten white balance, which creates a colour contrast between red and green tones. In this project, the goal is to recreate digitally the look of this specific film stock focusing on three aspects:

- the *film grain*: this film stock has a high sensitivity and produces a lot of grain, which is also pleasing aesthetically
- the *halo*: the halo is the main characteristic of this film stock
- the *colour profile*: the colour profile is not the one of your digital camera.

This project trains various image-to-image deep learning models on a novel dataset of image pairs capturing the same scene on a digital and film camera.

## 🔗 Shortcuts

Here is a list of things that you likely want to do:

- Check out the [demo](<>) of the model (*Not yet available*)
- Find all project details in the full [report](<>) (*Not yet available*)
- Inspect the experiment logs on [W&B](<>) (*Not yet available*)
- Download the image pair dataset from [Zenodo](<>) or [Huggingface](<>) (*Not yet available*)

## ⚙️ Installation

#### Pip

```bash
# clone project
git clone https://github.com/mikasenghaas/sillystill
cd sillystill

# [OPTIONAL] create conda environment
conda create -n sillystill python=3.9
conda activate sillystill

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/mikasenghaas/sillystill
cd stillystill

# create conda environment and install dependencies
conda env create -f environment.yaml -n sillystill

# activate conda environment
conda activate sillystill
```

#### Pre-Commit

Before contributing, please make sure that you install the Git hooks of `pre-commit` (e.g. it will be triggered on every git commit) by running the following command:

```bash
pre-commit install
```

To check if everything works as expected you can run `pre-commit run --all-files` which will run all hooks on the entire repository.

## 🤖 How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

## 🙌 Acknowledgements

This work was kindly funded by the [Image and Visual Representation (IVLR) Lab ](<>) at EPFL as part of the [Computational Photography (CS-413)](<>) class and supervised by [Raphaël Wan-Li Marc Achddou](https://people.epfl.ch/raphael.achddou) and [Sabine Süsstrunk](https://people.epfl.ch/sabine.susstrunk). Thank you for making this project possible!

This repository is bootstrapped from this [Lightning & Hydra Template](https://github.com/ashleve/lightning-hydra-template).
