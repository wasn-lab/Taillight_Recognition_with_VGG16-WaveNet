#!/bin/bash
set -x
set -e
IFS=$'\n'

readonly top_tmp_dir=/home/artifacts/weakness-detection
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly weakness_detection_dir=${repo_dir}/src/utilities/weakness_detection
readonly bags=$(find /mnt/backup_12T_1/Share/ADV/Rosbag/ -type f -name "*.bag")

if [[ -d /usr/local/lib/python3.6/dist-packages/cv2 ]]; then
  export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/dist-packages
fi

for bag_path in $bags; do
  hour=$(date "+%-H")
  while [[ "$hour" -gt 9 && "$hour" -lt 19 ]]; do
    # keep buildbot job alive
    echo "No weak detection during work hours."
    sleep 60
    hour=$(date "+%-H")
  done

  bag_basename=$(basename $bag_path)
  export TMP_DIR=${top_tmp_dir}/${bag_basename}
  if [[ ! -f ${TMP_DIR}/weak.tar.gz ]]; then
    set +e
    rosbag info "${bag_path}" | grep -i image
    if [[ "$?" == "0" ]]; then
      mkdir -p ${TMP_DIR}
      pushd ${weakness_detection_dir}
      bash main.sh "${bag_path}"
      popd
      pushd ${TMP_DIR}
      tar zcvf weak.tar.gz weak
      rm -r raw
      find ${TMP_DIR} -type f -exec chmod 644 {} \;
      find ${TMP_DIR} -type d -exec chmod 755 {} \;
      popd
      base_url="http://ci.itriadv.co/artifacts/weakness-detection/${bag_basename}"
      echo "visit ${base_url}/weak to get the results."
    else
      echo "${bag_path} does not contain images. Move to next bag."
    fi
    set -e
  fi
done
set +x
echo "All Done!"
