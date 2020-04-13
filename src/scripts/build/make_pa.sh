#! /bin/bash
cd /itriadv
sudo rm -r /build /devel
catkin_make -j --only-pkg camera_grabber