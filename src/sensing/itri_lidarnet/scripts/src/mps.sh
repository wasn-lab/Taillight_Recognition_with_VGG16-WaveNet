#!/bin/bash 
 
if [ "$(pidof nvidia-cuda-mps-control)" ] 
then
  echo 'MPS was found'
else
  gnome-terminal --tab -e 'bash -c "source $HOME/.bashrc;export CUDA_VISIBLE_DEVICES=0;echo itri | sudo -S nvidia-smi -i 0 -c EXCLUSIVE_PROCESS;echo itri | sudo -S nvidia-cuda-mps-control -f; exec bash"'
fi
