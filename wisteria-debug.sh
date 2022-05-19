#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-debug
#PJM -L gpu=4
#PJM -N Pytorch-VAE
#PJM -j
#PJM -m b
#PJM -m e
#PJM -o wisteria-logs/

# run commands
python3 run.py -c configs/vae-gpu.yaml