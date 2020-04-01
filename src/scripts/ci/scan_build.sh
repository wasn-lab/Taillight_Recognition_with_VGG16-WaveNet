#!/bin/bash
set -x
set -e
# Use debug mode to silence unused variables.
readonly build_type="${build_type:-Debug}"
readonly repo_dir=$(git rev-parse --show-toplevel)
readonly build_dir=build_scan_build
readonly devel_dir=devel_scan_build
if [[ -d /usr/local/llvm-6.0.0/bin ]]; then
  export PATH=/usr/local/llvm-6.0.0/bin:$PATH
fi
export CC=clang
export CXX=clang++

if [[ -d /var/www/html/scan_build ]]; then
  readonly output_dir=$(readlink -e /var/www/html/scan_build)
else
  readonly output_dir=/tmp
fi

pushd $repo_dir

# clean up the previous build.
for _dir in ${build_dir} ${devel_dir}; do
    if [[ -d $_dir ]]; then
        rm -rf $_dir
    fi
done
blacklist="dl_data"

scan-build -o ${output_dir} catkin_make \
    --build ${build_dir} \
    -DCATKIN_DEVEL_PREFIX=${devel_dir} \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DENABLE_CCACHE=0 \
    -DCAR_MODEL=OMNIBUS \
    -DCATKIN_BLACKLIST_PACKAGES="$blacklist" \
    -j6 ${EXTRA_CATKIN_ARGS}

# compress previous output
pushd ${output_dir}
readonly today=`date +"%Y-%m-%d"`
for d in `ls`; do
  if [[ -f $d ]]; then
    echo "Skip $d"
  elif [[ -d $d && "${d}" == "${today}"* ]]; then
    echo "Do not compress $d"
  else
    echo "Compress $d"
    tar cfJ ${d}.tar.xz $d
    rm -rf ${d}
  fi
done
popd

find ${output_dir} -type d -exec chmod 755 {} \;
find ${output_dir} -type f -exec chmod 644 {} \;
echo "Visit http://ci.itriadv.co/scan_build/ to see the html results (accessible in itri.org.tw only)."
popd
