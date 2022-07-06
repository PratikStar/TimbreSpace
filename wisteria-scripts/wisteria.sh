#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share
#PJM -L gpu=4
#PJM -N VQVAE
#PJM -j
#PJM -m b
#PJM -m e

# run commands
#python3 run-vanilla_vae.py -c configs/vae-gpu.yaml
python3 run-vq-vae.py