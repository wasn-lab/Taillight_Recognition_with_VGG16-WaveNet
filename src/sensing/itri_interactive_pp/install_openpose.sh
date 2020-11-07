#!/bin/bash
##########################################################
# for OpenCV 4 above                                     #
# openpose version: commit b365f48                       #
# caffe version: commit c95002fb                         #
# caffe version will depend on newest update on GitHub   #
##########################################################

sudo apt-get install protobuf-compiler libatlas-base-dev 

wget http://nas.itriadv.co:8888/git_data/B1/ped_models/openpose.tar.gz
tar zxvf openpose.tar.gz
cd ./openpose
cd ./3rdparty
git clone https://github.com/CMU-Perceptual-Computing-Lab/caffe.git
cd ..
mkdir build
cd ./build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DDOWNLOAD_BODY_25_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF -DDOWNLOAD_HAND_MODEL=OFF ..
make -j
sudo make install
cd ../..
rm openpose.tar.gz
