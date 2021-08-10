#!/bin/bash
set -x
pushd ~/itriadv
catkin_make -DCATKIN_WHITELIST_PACKAGES="pc2_compressor" -DCAR_MODEL=C2
popd
