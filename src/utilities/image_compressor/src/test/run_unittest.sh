#!/bin/bash
set -x
set -e
readonly repo_dir=$(git rev-parse --show-toplevel)
pushd ${repo_dir}/build
make -j image_compressor_test
${repo_dir}/devel/lib/image_compressor/image_compressor_test
popd
