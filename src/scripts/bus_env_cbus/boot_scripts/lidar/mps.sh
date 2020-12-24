#!/bin/bash
if [ "$(pidof nvidia-cuda-mps-control)" ];then
  echo 'MPS was found'
else
  export CUDA_VISIBLE_DEVICES=0
  nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
  nvidia-cuda-mps-control -f
fi
