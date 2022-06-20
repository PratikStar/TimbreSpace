#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-debug
#PJM -L gpu=4
#PJM -N Pytorch-VQVAE
#PJM -j
#PJM -m b
#PJM -m e

# run commands
python3 run-vae.py -c configs/vq_vae-gpu.yaml