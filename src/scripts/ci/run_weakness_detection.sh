#!/bin/bash
set -x
set -e

readonly top_tmp_dir=/home/artifacts/weakness-detection
export RSYNC_PASSWORD=itriu300
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly rsync_server="nas.itriadv.co"
readonly port=873
readonly path_prefix="Share/ADV/Rosbag/B1/"
readonly weakness_detection_dir=${repo_dir}/src/utilities/weakness_detection
readonly rsync_list=$(rsync -av --list-only --exclude test_case "rsync://icl_u300@${rsync_server}:${port}/${path_prefix}" | awk '{print $NF}')

if [[ -d /usr/local/lib/python3.6/dist-packages/cv2 ]]; then
  export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/dist-packages
fi

for bag_path in $rsync_list; do
  if [[ "$bag_path" == *".bag" ]]; then
    bag_basename=$(basename $bag_path)
    export TMP_DIR=${top_tmp_dir}/${bag_basename}
    bag_local_fullpath=${TMP_DIR}/${bag_basename}
    mkdir -p ${TMP_DIR}
    if [[ ! -f ${TMP_DIR}/weak.tar.gz ]]; then
      wget -c "http://nas.itriadv.co:8888/${path_prefix}${bag_path}" -O ${bag_local_fullpath}
      set +e
      rosbag info ${bag_local_fullpath} | grep -i image
      if [[ "$?" == "0" ]]; then
        pushd ${weakness_detection_dir}
        bash main.sh ${bag_local_fullpath}
        popd
        pushd ${TMP_DIR}
        tar zcvf weak.tar.gz weak
        rm -r raw
        popd
        rm ${bag_local_fullpath}
        base_url="http://ci.itriadv.co/artifacts/weakness-detection/${bag_basename}"
        echo "visit ${base_url}/weak to get the results."
      else
        echo "${bag_local_fullpath} does not contain images. Move to next bag."
      fi
      set -e
    fi
  fi
done
set +x
echo "All Done!"
