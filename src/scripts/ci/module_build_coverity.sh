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
readonly coverity_root=$(readlink -e $(dirname ${cov_build_bin})/..)
readonly cfgs="${coverity_root}/config/MISRA/MISRA_c2004_7.config ${coverity_root}/config/MISRA/MISRA_c2012_7.config ${coverity_root}/config/MISRA/MISRA_cpp2008_7.config
"
pushd $repo_dir

# clean up the previous build.
for _dir in build devel ${coverity_data_dir}; do
    if [[ -d $_dir ]]; then
        rm -rf $_dir
    fi
done

blacklist="localization;ndt_gpu"

cov-build --dir ${coverity_data_dir} --emit-complementary-info \
catkin_make -DENABLE_CCACHE=0 -DCATKIN_BLACKLIST_PACKAGES="$blacklist" ${EXTRA_CATKIN_ARGS} -j`nproc`

cov-analyze -dir ${coverity_data_dir} --strip-path ${repo_dir}/
for cfg in $cfgs; do
  cov-analyze --disable-default --misra-config ${cfg} --dir coverity_data --strip-path ${repo_dir}/
done

if [[ ! -z "${USER_NAME}" && ! -z "${USER_PASSWORD}" ]]; then
  cov-commit-defects --host 140.96.109.174 --dataport 9090 --stream master --dir coverity_data --user ${USER_NAME} --password ${USER_PASSWORD}
  echo "visit http://140.96.109.174:8080 to see the result (project: itriadv, stream: master)"
fi

popd
