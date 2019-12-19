#!/bin/bash
set -x
set -e

source devel/setup.bash
catkin_make run_tests_car_model_rostest_test_car_b1_drivenet_publish_detect_image.test

