#!/bin/bash
set -x
set -e

readonly repo_dir=$(git rev-parse --show-toplevel)

readonly cov_build_bin=$(which cov-build)
if [[ -z "${cov_build_bin}" ]]; then
  echo "Cannot find coverity. Please add cov_build into PATH"
  exit 0
fi

readonly coverity_data_dir="coverity_data"
readonly coverity_data_blacklist_dir="coverity_data_blacklist"
readonly coverity_root=$(readlink -e $(dirname ${cov_build_bin})/..)
readonly cfgs="${coverity_root}/config/MISRA/MISRA_c2004_7.config
               ${coverity_root}/config/MISRA/MISRA_c2012_7.config
               ${coverity_root}/config/MISRA/MISRA_cpp2008_7.config"
readonly blacklist="localization;ndt_gpu"

pushd $repo_dir

function cleanup {
  for _dir in build devel ${coverity_data_dir} ${coverity_data_blacklist_dir}; do
      if [[ -d $_dir ]]; then
          rm -rf $_dir
      fi
  done
}

function commit {
  data_dir=$1
  stream=$2
  if [[ ! -z "${USER_NAME}" && ! -z "${USER_PASSWORD}" ]]; then
    cov-commit-defects --host 140.96.109.174 --dataport 9090 --stream ${stream} --dir ${data_dir} --user ${USER_NAME} --password ${USER_PASSWORD}
  fi
}

function analyze {
  data_dir=$1
  cov-analyze -dir ${data_dir} --strip-path ${repo_dir}/
  for cfg in $cfgs; do
    cov-analyze --disable-default --misra-config ${cfg} --dir ${data_dir} --strip-path ${repo_dir}/
  done
}

# build blacklisted packages
cleanup
cov-build --dir ${coverity_data_blacklist_dir} --emit-complementary-info \
  catkin_make -DENABLE_CCACHE=0 -DCATKIN_WHITELIST_PACKAGES="${blacklist};cuda_downsample" ${EXTRA_CATKIN_ARGS}
analyze ${coverity_data_blacklist_dir}
commit ${coverity_data_blacklist_dir} master

# build all except blacklisted packages
# Use this order because most people can see analysis result at the most recent stream.
cleanup
cov-build --dir ${coverity_data_dir} --emit-complementary-info \
  catkin_make -DENABLE_CCACHE=0 -DCATKIN_BLACKLIST_PACKAGES="$blacklist" ${EXTRA_CATKIN_ARGS}
analyze ${coverity_data_dir}
commit ${coverity_data_dir} localization

echo "visit http://140.96.109.174:8080 to see the result (project: itriadv, stream: master)"
popd
