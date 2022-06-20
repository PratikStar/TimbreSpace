#!/bin/bash
#PJM -g gk77
#PJM -L rscgrp=share-interactive
#PJM -N Pytorch-VAE
#PJM -j
#PJM --interact

echo "Hostname: $HOSTNAME"
echo "pwd: $(pwd)"

echo "====== CPU info ======"
lscpu
echo "======================"

echo "====== GPU info ======"
nvidia-smi
echo "======================"

echo "Python path: $(which python3)"
echo "Python version: $(python3 --version)"

echo "====== pip list ======"
pip3 list
echo "======================"
