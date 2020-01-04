#!/bin/bash

sudo apt-get install protobuf-compiler libatlas-base-dev 
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
cd ./openpose
git checkout v1.5.1
mkdir build
cd ./build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF ..
make -j
sudo make install

