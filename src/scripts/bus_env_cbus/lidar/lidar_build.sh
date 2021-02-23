#!/bin/bash
set -x

catkin_make -DCMAKE_BUILD_TYPE=Release -DCAR_MODEL=C1
