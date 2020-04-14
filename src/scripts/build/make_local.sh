#! /bin/bash
cd /itriadv
sudo rm -r /build /devel
catkin_make -j -DCATKIN_WHITELIST_PACKAGES="msgs;map_pub;localization;cuda_downsample;ndt_gpu"