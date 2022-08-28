#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-interactive
#PJM -N something
#PJM -j
#PJM --interact

# install dependencies
which python3
which pip3
which python
echo "======="

#pip3 list
python run-generic.py

#python3 -m venv timbre
#source ./timbre/bin/activate
#which python3
#which pip3
##pip3 install wandb==0.12.21
#pip3 install -r requirements-gpu.txt