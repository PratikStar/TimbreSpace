#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-short
#PJM -L gpu=4
#PJM -N TimbreTransfer
#PJM -j
#PJM -m b
#PJM -m e

# run commands
python run-generic.py -c configs/timbre_transfer.yaml
