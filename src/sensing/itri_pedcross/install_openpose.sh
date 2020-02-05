#!/bin/bash

sudo apt-get install protobuf-compiler libatlas-base-dev 
# openpose version: commit 825f0d0
# caffe version: commit b5ede48
wget http://nas-cht.itriadv.co:8888/Share/ADV/S3_git_data/openpose.tar.gz
tar zxvf openpose.tar.gz
cd ./openpose
mkdir build
cd ./build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF ..
make -j
sudo make install
cd ../..
rm openpose.tar.gz
