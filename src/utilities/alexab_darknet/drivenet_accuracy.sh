#!/bin/bash
set -x
set -e

readonly repo_dir=$(git rev-parse --show-toplevel)
readonly darknet_dir=$(dirname $(readlink -e $0))
readonly drivenet_dir=${repo_dir}/src/sensing/itri_drivenet/drivenet
readonly cfg_file=${drivenet_dir}/data/yolo/yolov3.cfg


if [[ "$1" == "fov60" ]]; then
  readonly testset_dir=${darknet_dir}/drivenet_dataset/Dataset_Fov60
  readonly data_file=cfg/drivenet_fov60.data
  readonly weights_file=${drivenet_dir}/data/yolo/yolov3_b1.weights
  readonly log_file=drivenet_fov60.log
elif [[ "$1" == "fov120" ]]; then
  readonly testset_dir=${darknet_dir}/drivenet_dataset/Dataset_Fov120
  readonly data_file=cfg/drivenet_fov120.data
  readonly weights_file=${drivenet_dir}/data/yolo/yolov3_fov120_b1.weights
  readonly log_file=drivenet_fov120.log
elif [[ "$1" == "all" ]]; then
  readonly testset_dir=${darknet_dir}/drivenet_dataset
  readonly data_file=cfg/drivenet.data
  readonly weights_file=${drivenet_dir}/data/yolo/yolov3_b1.weights
  readonly log_file=drivenet.log
else
  echo "Usage: $0 fov60|fov120|all"
  exit 1
fi

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
  python gen_testset_list.py --testset-dir ${testset_dir}
  build/darknet detector map ${data_file} ${cfg_file} ${weights_file} | tee ${log_file}
}

build_darknet_exe
calc_accuracy
if [[ "$1" == "all" ]]; then
  python post_accuracy.py --log-file ${log_file}
fi

echo "All done!"

popd
