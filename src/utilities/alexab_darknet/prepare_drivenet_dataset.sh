#!/bin/bash
set -x
set -e

export RSYNC_PASSWORD=itriu300
readonly port=873
readonly darknet_dir=$(dirname $(readlink -e $0))
pushd ${darknet_dir}

mkdir -p drivenet_dataset
pushd drivenet_dataset

rsync -av --delete "rsync://icl_u300@nas-cht.itriadv.co:${port}/Share/ADV/Camera Models/S3/DriveNet/Dataset/Dataset_Fov60" .
rsync -av --delete "rsync://icl_u300@nas-cht.itriadv.co:${port}/Share/ADV/Camera Models/S3/DriveNet/Dataset/Dataset_Fov120" .
#lftp -u icl_u300,itriu300 ftp://nas-cht.itriadv.co -e "cd /Share/ADV/Camera\ Models/S3/DriveNet/Dataset; mirror -c Dataset_Fov120; mirror -c Dataset_Fov60; bye"

popd
popd
