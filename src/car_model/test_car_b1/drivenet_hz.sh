#!/bin/bash
set -x
set -e

source devel/setup.bash
roslaunch car_model drivenet_hz.test

