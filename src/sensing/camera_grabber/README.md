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

**CAR_MODEL = B1_V3:**
```
catkin_make -DCATKIN_WHITELIST_PACKAGES="msgs;camera_utils;car_model;camera_grabber;dl_data" -DCAR_MODEL=B1_V3
```

**CAR_MODEL = C1:**
```
catkin_make -DCATKIN_WHITELIST_PACKAGES="msgs;camera_utils;car_model;camera_grabber;dl_data" -DCAR_MODEL=C1
```

### How to run


**CAR_MODEL = B1_V3 or C1:**

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

4. car_driver 
```
true  : use car mode camera driver (default)
false : use laboratory mode camera driver
```

5. motion_vector 
```
true  : enable motion vector message (The do_resize have to false when motion_vector true)
        motion vector message topic names are :
        /cam/front_bottom_60/motion_vector_msg
        /cam/front_top_far_30/motion_vector_msg
        /cam/front_top_close_120/motion_vector_msg
        /cam/right_front_60/motion_vector_msg
        /cam/right_back_60/motion_vector_msg
        /cam/left_front_60/motion_vector_msg
        /cam/left_back_60/motion_vector_msg
        /cam/back_top_120/motion_vector_msg
false : disable motion vector message (default)
```

### Motion vector image debug flag
```
If you want to debug motion vector image. There is a line "//#define MV_IMAGE_DEBUG 1" in the JetsonXavierGrabber.cpp. 
It can publish motion vector image when unmark the line.
The do_resize set false and motion_vector set true at the same time.
Motion vector image topic names are :
/cam/front_bottom_60/motion_vector
/cam/front_top_far_30/motion_vector
/cam/front_top_close_120/motion_vector
/cam/right_front_60/motion_vector
/cam/right_back_60/motion_vector
/cam/left_front_60/motion_vector
/cam/left_back_60/motion_vector
/cam/back_top_120/motion_vector
```

### How to evaluate results


**CAR_MODEL = B1_V3 or C1:**

```
use rqt_image_view to view following topic
/cam/front_bottom_60/raw
/cam/front_top_far_30/raw
/cam/front_top_close_120/raw
/cam/right_front_60/raw
/cam/right_back_60/raw
/cam/left_front_60/raw
/cam/left_back_60/raw
/cam/back_top_120/raw
```

### How to install camera driver (CAR_MODEL = B1_V3)

**install camera driver for car mode**
```
bash /home/nvidia/itriadv/src/scripts/bus_env/xavier/init_camera_driver_for_car_mode.sh
```

**install camera driver for laboratory mode**
```
bash /home/nvidia/itriadv/src/scripts/bus_env/xavier/init_camera_driver_for_laboratory_mode.sh
```

### How to install camera driver when xavier bootup (CAR_MODEL = B1_V3)
**install camera driver for car mode**
```
bash /home/nvidia/itriadv/src/scripts/bus_env/xavier/init_car_mode_camera_driver_to_bootup_script.sh
```

**install camera driver for laboratory mode**
```
bash /home/nvidia/itriadv/src/scripts/bus_env/xavier/init_laboratory_mode_camera_driver_to_bootup_script.sh
```


### How to install camera driver (CAR_MODEL = C1)

**install camera driver for car mode**
```
bash /home/nvidia/itriadv/src/scripts/bus_env_cbus/xavier/init_camera_driver_for_car_mode.sh
```

**install camera driver for laboratory mode**
```
bash /home/nvidia/itriadv/src/scripts/bus_env_cbus/xavier/init_camera_driver_for_laboratory_mode.sh
```

### How to install camera driver when xavier bootup (CAR_MODEL = C1)
**install camera driver for car mode**
```
bash /home/nvidia/itriadv/src/scripts/bus_env_cbus/xavier/init_car_mode_camera_driver_to_bootup_script.sh
```

**install camera driver for laboratory mode**
```
bash /home/nvidia/itriadv/src/scripts/bus_env_cbus/xavier/init_laboratory_mode_camera_driver_to_bootup_script.sh
```



