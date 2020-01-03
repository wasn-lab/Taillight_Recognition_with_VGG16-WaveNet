#!/bin/bash

sudo apt-get install protobuf-compiler libatlas-base-dev 
#git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
wget https://github.com/CMU-Perceptual-Computing-Lab/openpose/archive/v1.5.1.tar.gz -O openpose-1.5.1.tar.gz
cd ./openpose
wget http://posefs1.perception.cs.cmu.edu/OpenPose/3rdparty/windows/caffe_15_2019_05_16.zip
unzip caffe_15_2019_05_16.zip
mv caffe 3rdparty
mkdir build
cd ./build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/openpose -DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF ..
make -j`nproc`
sudo make install

