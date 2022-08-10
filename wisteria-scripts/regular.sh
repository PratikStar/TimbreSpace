#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -N TimbreTransferFlatten
#PJM -j
#PJM -m b
#PJM -m e

# run commands
python run-generic.py -c configs/timbre_transfer_flatten.yaml
