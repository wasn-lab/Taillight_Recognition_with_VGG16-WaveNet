#!/bin/bash 
if [ "$(pidof nvidia-cuda-mps-control)" ];then
	gnome-terminal --tab -e  'bash -c "sudo_pw=itri; source $HOME/.bashrc;export CUDA_VISIBLE_DEVICES=0; echo $sudo_pw | sudo -S nvidia-smi -i 0 -c DEFAULT; echo $sudo_pw | echo quit | sudo nvidia-cuda-mps-control; exec bash"'
  echo 'Close the mps.'
else
  echo 'MPS is not found'
fi
