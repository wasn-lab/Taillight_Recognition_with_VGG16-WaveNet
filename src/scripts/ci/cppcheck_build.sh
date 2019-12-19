#!/bin/bash
set -x
set -e

export PATH=/usr/local/cmake-3.15.5/bin:$PATH
echo "The script works only for cmake >= 3.10. Your cmake version is:"
cmake --version

readonly build_type="${build_type:-Debug}"
readonly repo_dir=$(git rev-parse --show-toplevel)
pushd $repo_dir

# clean up the previous build.
for _dir in build ; do
    if [[ -d $_dir ]]; then
        rm -rf $_dir
    fi
done

catkin_make \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCATKIN_BLACKLIST_PACKAGES="dl_data;localization;opengl_test" \
    -DCMAKE_CXX_CPPCHECK="cppcheck;--template=gcc;--enable=style,warning;--inconclusive;--inline-suppr;--suppressions-list=${repo_dir}/src/scripts/ci/cppcheck_suppression.txt" ${EXTRA_CATKIN_ARGS}
popd
