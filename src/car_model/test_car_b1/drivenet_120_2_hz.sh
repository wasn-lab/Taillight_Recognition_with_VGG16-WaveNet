#!/bin/bash
set -x
set -e

source devel/setup.bash
roslaunch car_model drivenet_120_2_hz.test

