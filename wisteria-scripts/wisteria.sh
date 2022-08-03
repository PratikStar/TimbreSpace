#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-short
#PJM -L gpu=4
#PJM -N MusicVAEFlat
#PJM -j
#PJM -m b
#PJM -m e

# run commands
python run-timbre-vae.py -c configs/music_vae-gpu.yaml