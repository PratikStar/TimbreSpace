#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share
#PJM -L gpu=4
#PJM -N MusicTimbreVAE
#PJM -j
#PJM -m b
#PJM -m e

# run commands
python run-timbre-vae.py -c configs/timbre_vae-gpu.yaml