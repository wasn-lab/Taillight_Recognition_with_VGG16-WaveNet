#!/bin/bash
source /home/lidar/itriadv/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.3:11311

roslaunch --wait fail_safe pad_issue_reporter.launch
