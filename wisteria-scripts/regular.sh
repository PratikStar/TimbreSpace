#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -N CondConv
#PJM -j
#PJM -m b
#PJM -m e

# run commands
source /work/01/gk77/k77021/.bashrc
export HOME=/work/01/gk77/k77021
#env
#python run-generic.py -c configs/timbre_transfer.yaml "wandb.project=condconv"
wandb agent auditory-grounding/condconv/5daxpq5n