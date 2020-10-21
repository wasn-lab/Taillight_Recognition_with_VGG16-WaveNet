### Install camera_grabber dependencies
**Jetson Xavier**

* Install JetPack 4.4  (JetPack 4.4 contain CUDA 10.2 , TensorRT 7.1.3 )
* Install ROS
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt install ros-melodic-desktop-full
```
* Install package
```
sudo apt-get install libgoogle-glog-dev
sudo apt-get install libglm-dev 
sudo apt install ros-melodic-grid-map
```
* create opencv symbolic link
```
cd /usr/include
sudo ln -s opencv4/ opencv
```
### Compile 


```
catkin_make -DCATKIN_WHITELIST_PACKAGES="msgs;camera_utils;car_model;camera_grabber;dl_data" -DCAR_MODEL=B1_V3
```

### How to run


**CAR_MODEL = B1_V3:**

```
source ./devel/setup.bash
```

/cam/front_bottom_60, /cam/front_top_far_30, /cam/front_top_close_120, /cam/right_front_60
/cam/right_back_60,  /cam/left_front_60, /cam/left_back_60, /cam/back_top_120
```
roslaunch camera_grabber jetson_xavier_b1.launch
```


### How to setup parameters of launch file


1. expected_fps
```
22 (default)

```

2. do_resize
```
true  : enable resize (default)
false : disable resize
```

3. password (It need 'sudo' password when initiate camera driver.)
```
nvidia (default)
```


### How to evaluate results


**CAR_MODEL = B1_V3:**

```
use rqt_image_view to view following topic
/cam/front_bottom_60
/cam/front_top_far_30
/cam/front_top_close_120
/cam/right_front_60
/cam/right_back_60
/cam/left_front_60
/cam/left_back_60
/cam/back_top_120
```
