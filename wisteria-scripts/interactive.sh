#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-interactive
#PJM -N TimbreTransfer
#PJM -j

# run commands
#python run-generic.py -c configs/timbre_transfer_flatten.yaml

source /work/01/gk77/k77021/.bashrc
echo "loaded source"
export HOME=/work/01/gk77/k77021
echo "$HOME"

#pip3 install pytorch-lightning==1.5.6

#pip3 install -r requirements-gpu.txt
wandb agent auditory-grounding/condconv/mvbjp81s