#!/bin/bash 
if [ "$(pidof nvidia-cuda-mps-control)" ];then
  echo 'MPS was found'
else
   gnome-terminal --tab -e  'bash -c "hostname; sudo_pw=amyeu300; source $HOME/.bashrc;export CUDA_VISIBLE_DEVICES=0; echo $sudo_pw | sudo -S nvidia-smi -i 0 -c EXCLUSIVE_PROCESS; echo sudo_pw | sudo -S nvidia-cuda-mps-control -f; exec bash"'
  echo 'MPS is ready'
fi
