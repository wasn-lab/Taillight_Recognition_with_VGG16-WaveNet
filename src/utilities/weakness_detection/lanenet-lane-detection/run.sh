#!/bin/bash
set -x
set -e

readonly lanenet_dir=$(dirname $(readlink -e $0))
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/dist-packages
export PYTHONPATH=$PYTHONPATH:${lanenet_dir}

function dl_data {
  if [[ ! -f ${lanenet_dir}/data/tusimple_ipm_remap.yml ]]; then
    wget http://nas.itriadv.co:8888/git_data/B1/lanenet/tusimple_ipm_remap.yml -O ${lanenet_dir}/data/tusimple_ipm_remap.yml
  fi

  if [[ ! -f ${lanenet_dir}/bisenet-v2/tusimple_lanenet.ckpt.data-00000-of-00001 ]]; then
    wget http://nas.itriadv.co:8888/git_data/B1/lanenet/bisenet-v2/tusimple_lanenet.ckpt.data-00000-of-00001 -O ${lanenet_dir}/bisenet-v2/tusimple_lanenet.ckpt.data-00000-of-00001
    wget http://nas.itriadv.co:8888/git_data/B1/lanenet/bisenet-v2/tusimple_lanenet.ckpt.index -O ${lanenet_dir}/bisenet-v2/tusimple_lanenet.ckpt.index
    wget http://nas.itriadv.co:8888/git_data/B1/lanenet/bisenet-v2/tusimple_lanenet.ckpt.meta -O ${lanenet_dir}/bisenet-v2/tusimple_lanenet.ckpt.meta
  fi
}

dl_data
python find_lanes.py --weights-path ${lanenet_dir}/bisenet-v2/tusimple_lanenet.ckpt --image-filenames /home/chtseng/repo/itriadv/src/utilities/alexab_darknet/drivenet_dataset/Dataset_Fov_raw/one_valid.txt
