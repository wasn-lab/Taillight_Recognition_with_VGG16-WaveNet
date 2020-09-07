#!/bin/bash
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly wdir=${repo_dir}/src/utilities/alexab_darknet/drivenet_weights_mr_test/fov60
readonly base_url=http://nas.itriadv.co:8888/git_data/B1/drivenet_weights_mr_test/fov60

pushd $wdir
for jfile in `ls *.json`; do
  bn=$(basename $jfile .json)
  dest=${bn}.jpg
  if [[ ! -f $dest ]]; then
    wget ${base_url}/${dest}
  else
    echo "Skip ${dest} as it has been downloaded."
  fi
done

popd
