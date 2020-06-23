#!/bin/bash
set -x
set -e
TMP_DIR=/tmp
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly darknet_dir=${repo_dir}/src/utilities/alexab_darknet
readonly drivenet_dir=${repo_dir}/src/sensing/itri_drivenet/drivenet
readonly weakness_det_dir=${repo_dir}/src/utilities/weakness_detection
readonly cfg_file=${drivenet_dir}/data/yolo/yolov3.cfg
readonly data_file=${darknet_dir}/cfg/drivenet_fov60.data
readonly weights_file=${drivenet_dir}/data/yolo/yolov3_b1.weights
readonly cam_ids="back_top_120 front_bottom_60 front_top_close_120 front_top_far_30 left_back_60 left_front_60 right_back_60 right_front_60"
readonly image_list_txt=${TMP_DIR}/weakness_detection_image_list.txt
readonly yolo_result_json=${TMP_DIR}/yolo_result.json

source $repo_dir/devel/setup.bash

# check if ROS master is alive
rostopic list

function make_itrisaver {
  pushd $repo_dir/build
  make -j image_saver deeplab_cmd
  popd
}

function save_images {
  set +e
  for cam_id in $cam_ids; do
    topic="/cam/${cam_id}"
    output_dir=${TMP_DIR}/${cam_id}
    mkdir -p $output_dir
    ${repo_dir}/devel/lib/itri_file_saver/image_saver -image_topic ${topic} -output_dir ${output_dir} 2>/dev/null &
  done
  sleep 3

  if [[ "$#" == "0" ]]; then
    echo "Usage: $0 bag1 [bag2] ..."
    exit 1
  fi
  for bag in $@; do
    if [[ "$bag" == *".bag" ]]; then
      rosbag play $bag
    fi
  done

  sleep 3  # wait for savers finish their jobs
  killall image_saver
  set -e
  true > ${image_list_txt}
  for cam_id in $cam_ids; do
    output_dir=${TMP_DIR}/${cam_id}
    find $output_dir -name "*.jpg" -type f | grep -v yolo | grep -v deeplab >> ${image_list_txt}
  done
}

function rm_tmp_files {
  for cam_id in $cam_ids; do
    output_dir=${TMP_DIR}/${cam_id}
    if [[ -d ${output_dir} ]]; then
      rm -r $output_dir
    fi
  done
  if [[ -f ${image_list_txt} ]]; then
    rm ${image_list_txt}
  fi
}

function run_deeplab {
  ${repo_dir}/devel/lib/deeplab/deeplab_cmd < ${image_list_txt}
}

function build_darknet_exe {
  pushd ${darknet_dir}
  if [[ -d build ]]; then
    rm -r build
  fi

  mkdir -p build
  pushd build
  cmake .. -DENABLE_OPENCV=0
  make -j
  popd
  popd
}

function run_yolo {
  python3 gen_yolo_detection_img.py
  if [[ -f ${darknet_dir}/predictions.jpg ]]; then
    rm ${darknet_dir}/predictions.jpg
  fi
}

function find_weakness {
  python find_weakness.py
}

function run_efficientdet {
  pushd ${weakness_det_dir}
  if [[ ! -f Yet-Another-EfficientDet-Pytorch/weights/efficientdet-d4.pth ]]; then
    mkdir -p Yet-Another-EfficientDet-Pytorch/weights
    pushd Yet-Another-EfficientDet-Pytorch/weights
    wget http://nas.itriadv.co:8888/git_data/B1/efficientdet-pytorch/efficientdet-d4.pth
    popd
  fi
  source ~/py36_efficientdet/bin/activate
  pushd Yet-Another-EfficientDet-Pytorch
  python efficientdet_itri.py
  deactivate
  popd
  popd
}

make_itrisaver
save_images $@
run_deeplab
build_darknet_exe
run_yolo
run_efficientdet

find_weakness
#rm_tmp_files
