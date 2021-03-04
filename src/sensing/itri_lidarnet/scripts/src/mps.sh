#!/bin/bash 
if [ "$(pidof nvidia-cuda-mps-control)" ];then
  echo 'MPS was found'
else
	case $HOSTNAME in
    (lidar) gnome-terminal --tab -e  'bash -c "hostname; sudo_pw=itri; source $HOME/.bashrc;export CUDA_VISIBLE_DEVICES=0; echo $sudo_pw | sudo -S nvidia-smi -i 0 -c EXCLUSIVE_PROCESS; echo sudo_pw | sudo -S nvidia-cuda-mps-control -f; exec bash"';;
    (camera) gnome-terminal --tab -e  'bash -c "hostname; sudo_pw=itri; source $HOME/.bashrc;export CUDA_VISIBLE_DEVICES=0; echo $sudo_pw | sudo -S nvidia-smi -i 0 -c EXCLUSIVE_PROCESS; echo sudo_pw | sudo -S nvidia-cuda-mps-control -f; exec bash"';;
    (*)   gnome-terminal --tab -e  'bash -c "hostname; sudo_pw=123456; source $HOME/.bashrc;export CUDA_VISIBLE_DEVICES=0; echo $sudo_pw | sudo -S nvidia-smi -i 0 -c EXCLUSIVE_PROCESS; echo sudo_pw | sudo -S nvidia-cuda-mps-control -f; exec bash"';;
	esac
fi
