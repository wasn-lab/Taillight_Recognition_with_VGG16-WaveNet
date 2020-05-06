#!/bin/bash
set -x
set -e

readonly darknet_dir=$(dirname $(readlink -e $0))
pushd ${darknet_dir}

mkdir -p drivenet_dataset
pushd drivenet_dataset

lftp -u icl_u300,itriu300 ftp://nas-cht.itriadv.co -e "cd /Share/ADV/Camera\ Models/S3/DriveNet/Dataset; mirror -c Dataset_Fov120; mirror -c Dataset_Fov60; bye"

popd
popd
