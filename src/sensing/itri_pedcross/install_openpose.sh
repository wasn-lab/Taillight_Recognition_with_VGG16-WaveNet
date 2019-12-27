#!/bin/bash

git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
cd ./openpose
mkdir build
cd ./build
cmake -DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF ..
make -j`nproc`
sudo make install

