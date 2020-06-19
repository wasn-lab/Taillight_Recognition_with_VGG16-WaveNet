#!/bin/bash
set -x
set -e

export RSYNC_PASSWORD=itriu300
rsync_server="nas.itriadv.co"
readonly port=873
readonly darknet_dir=$(dirname $(readlink -e $0))

pushd ${darknet_dir}
mkdir -p drivenet_dataset
pushd drivenet_dataset

rsync -av --delete "rsync://icl_u300@${rsync_server}:${port}/Dataset/DriveNet/Dataset/Dataset_Fov60" .
rsync -av --delete "rsync://icl_u300@${rsync_server}:${port}/Dataset/Drivenet/Dataset_Fov120" .

popd
popd
