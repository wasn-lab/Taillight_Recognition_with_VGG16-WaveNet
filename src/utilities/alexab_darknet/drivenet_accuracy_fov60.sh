#!/bin/bash
set -x
set -e

readonly repo_dir=$(git rev-parse --show-toplevel)
readonly darknet_dir=$(dirname $(readlink -e $0))
readonly drivenet_dir=${repo_dir}/src/sensing/itri_drivenet/drivenet

pushd ${darknet_dir}

function build_darknet_exe {
  if [[ ! -d build ]]; then
    mkdir -p build
    pushd build
    cmake .. -DENABLE_OPENCV=0
    make -j
    popd
  fi
}

function calc_accuracy {
  testset_dir=${darknet_dir}/drivenet_dataset/Dataset_Fov60
  weights_file=${drivenet_dir}/data/yolo/yolov3_b1.weights
  cfg_file=${drivenet_dir}/data/yolo/yolov3.cfg
  python gen_testset_list.py --testset-dir ${testset_dir}
  build/darknet detector map cfg/drivenet_fov60.data ${cfg_file} ${weights_file}
}

build_darknet_exe
calc_accuracy

echo "All done!"

popd
