#!/bin/bash
set -x
set -e
readonly today=$(date "+%Y%m%d")

pushd build
make video_saver -j
popd

source build/car_model/scripts/car_model.sh
if [[ "${CAR_MODEL}" != "B1_V2" ]]; then
  echo "This script is for B1_V2 only."
  exit 0
fi

source devel/setup.bash
export output_dir=/var/www/html/artifacts/drivenet-detection/${today}
mkdir -p $output_dir
chmod 775 $output_dir

roslaunch car_model drivenet_output_as_video.test

pushd $output_dir
for avi in `ls *.avi`; do
  webm=$(basename $avi .avi).webm
  ffmpeg -y -i ${avi} $webm
done
popd

cp src/scripts/ci/show_drivenet_videos.html $output_dir
find $output_dir -type f -exec chmod 644 {} \;

set +x
echo "Visit http://ci.itriadv.co/artifacts/drivenet-detection/${today} to download the results."
