#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-interactive
#PJM -N Pytorch-VAE
#PJM -j
#PJM --interact

# install dependencies
#pip3 install -r requirements.txt
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
