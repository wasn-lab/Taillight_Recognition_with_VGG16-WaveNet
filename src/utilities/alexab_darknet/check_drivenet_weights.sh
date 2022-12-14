#!/bin/bash
exit 0
set -x
set -e
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly commit_id=$(git rev-parse HEAD)
readonly branch_name=$(git rev-parse --abbrev-ref HEAD)
readonly darknet_dir=$(dirname $(readlink -e $0))
readonly drivenet_dir=${repo_dir}/src/sensing/itri_drivenet/drivenet
readonly cfg_file=${drivenet_dir}/data/yolo/yolov3.cfg
readonly weakness_detection_dir=${repo_dir}/src/utilities/weakness_detection
readonly now=$(date "+%Y%m%d%H%M%S")
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/dist-packages
set +e
git fetch
git merge remotes/origin/master --no-ff
git diff-index --quiet HEAD
if [[ "$?" == "0" ]]; then
  readonly repo_status="clean"
else
  readonly repo_status="dirty"
fi
set -e

if [[ "${USER}" == "icl_u300" ]]; then
  readonly artifacts_dir=/home/artifacts/drivenet-weights-check/${now}
else
  readonly artifacts_dir=/tmp/${now}
fi

readonly data_file_fov60=cfg/drivenet_fov60.data
readonly weights_fov60=${drivenet_dir}/data/yolo/yolov3_b1.weights
readonly data_file_fov120=cfg/drivenet_fov120.data
readonly weights_fov120=${drivenet_dir}/data/yolo/yolov3_fov120_b1.weights


function build_darknet_exe {
  if [[ ! -d build ]]; then
    mkdir -p build
    pushd build
    cmake .. -DENABLE_OPENCV=0
    make -j
    popd
  fi
}

function dl_drivenet_weights {
  if [[ -f ${weights_fov60} ]]; then
    rm ${weights_fov60}
  fi
  wget http://nas.itriadv.co:8888/git_data/B1/drivenet/yolov3_b1.weights -O ${weights_fov60}
  if [[ -f ${weights_fov120} ]]; then
    rm ${weights_fov120}
  fi
  wget http://nas.itriadv.co:8888/git_data/B1/drivenet/yolov3_fov120_b1.weights -O ${weights_fov120}
  md5sum ${weights_fov60} ${weights_fov120}
}

function mr_test {
  if [[ "$1" == "" ]]; then
    angle="fov60"
  else
    angle=$1
  fi
  readonly dest_dir=${artifacts_dir}/${angle}
  readonly src_dir=${darknet_dir}/drivenet_weights_mr_test/${angle}
  mkdir -p ${dest_dir}

  find ${src_dir} -name "*.jpg" -exec cp {} ${dest_dir} \;
  find ${src_dir} -name "*.json" -exec cp {} ${dest_dir} \;
  find ${dest_dir} -name "*.jpg" | grep -v _yolo.jpg | grep -v _expect.jpg > ${dest_dir}/mr_images.txt
  python3 ${weakness_detection_dir}/gen_yolo_detection_img.py --yolo-result-json ${dest_dir}/yolo_result.json --image-filenames ${dest_dir}/mr_images.txt
  for image_filename in `cat ${dest_dir}/mr_images.txt`; do
    python3 ${darknet_dir}/drivenet_weights_mr_test/draw_bbox.py -i ${image_filename} --output-dir ${dest_dir}
  done
  rm ${dest_dir}/mr_images.txt
  python3 drivenet_weights_mr_test/check_detection_result.py --yolo-result-json ${dest_dir}/yolo_result.json --output-dir ${dest_dir}
}

mkdir -p ${artifacts_dir}

pushd ${darknet_dir}
build_darknet_exe
dl_drivenet_weights
bash drivenet_weights_mr_test/fov60/dl_jpg.sh
mr_test fov60
set +e
python3 drivenet_weights_mr_test/merge_fov60_120_result.py --artifacts-dir ${artifacts_dir} --branch-name ${branch_name} --commit-id ${commit_id} --repo-status ${repo_status}
final_result=$?
python3 drivenet_weights_mr_test/post_to_backend.py --json-result ${artifacts_dir}/check_result.json
find ${artifacts_dir} -type d -exec chmod 755 {} \;
find ${artifacts_dir} -type f -exec chmod 644 {} \;
set +x
echo "All done!"
echo "Artifacts can be bound in http://ci.itriadv.co/artifacts/drivenet-weights-check/${now}"
popd
exit $final_result
