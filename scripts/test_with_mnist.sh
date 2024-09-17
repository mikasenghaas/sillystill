#!/bin/bash

python src/train.py data=mnist_identity 'model.net.features=[16, 32, 64]'