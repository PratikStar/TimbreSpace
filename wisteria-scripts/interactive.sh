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
#env
python run-generic.py -c configs/timbre_transfer.yaml "trainer_params.max_epochs=2"
#wandb agent auditory-grounding/try/oy7fg2zg