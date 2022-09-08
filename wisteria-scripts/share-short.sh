#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-short
#PJM -L gpu=4
#PJM -N CondConv
#PJM -j
#PJM -m b
#PJM -m e

# run commands
source /work/01/gk77/k77021/.bashrc
export HOME=/work/01/gk77/k77021
#env
wandb agent auditory-grounding/condconv/zf4xlgxz
#python run-generic.py -c configs/timbre_transfer.yaml "wandb.project=condconv" "trainer_params.max_epochs=5"

