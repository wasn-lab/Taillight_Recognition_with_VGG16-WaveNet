### Install drivenet dependencies

* Install Cuda 10.0
* Install TensorRT 5
* Install PCL 1.9.1

* Install ros-kinetic-cv-bridge, ros-kinetic-opencv3 and image-transport
```
sudo apt-get install ros-kinetic-image-transport
sudo apt-get install ros-kinetic-cv-bridge
```

### Compile (Important!!!)

Please choose Release mode.

If not, the module will run slowly.

```
catkin_make -DCMAKE_BUILD_TYPE=Release
```

### How to run

**CAR_MODEL = B1:**

```
source ./devel/setup.bash
```

FOV60 CamObjFrontRight, CamObjFrontCenter, CamObjFrontLeft  2D object detection
```
roslaunch drivenet b1_drivenet_60.launch
```

FOV120 CamObjRightFront, CamObjRightBack, CamObjLeftFront, CamObjLeftBack  2D object detection
```
roslaunch drivenet b1_drivenet_120_1.launch
```

FOV120 CamObjFrontCenter, CamObjBackTop  2D object detection
```
roslaunch drivenet b1_drivenet_120_2.launch
```

Camera & LiDAR 3D object detection
```
roslaunch alignment b1_3d_object_detection.launch
```

**CAR_MODEL = B1_V2:**

```
source ./devel/setup.bash
```

/cam/front_bottom_60, /cam/front_top_far_30 2D object detection
```
roslaunch drivenet b1_v2_drivenet_group_a.launch
```

/cam/front_top_close_120, /cam/right_front_60, /cam/right_back_60  2D object detection
```
roslaunch drivenet b1_v2_drivenet_group_b.launch
```

/cam/left_front_60, /cam/left_back_60, /cam/back_top_120  2D object detection
```
roslaunch drivenet b1_v2_drivenet_group_c.launch
```

Camera & LiDAR 3D object detection
```
roslaunch alignment b1_v2_3d_object_detection.launch
```

### How to setup parameters of launch file

1. car_id (Choose car config)
```
1: car 1 (b1~.launch)
```

1. standard_fps (regular publisher)
```
0: disable (default)
1: enable
```

2. display  (2D detection visualization)
```
0: disable (default)
1: enable
```

3. input_resize (2D detection visualization)
```
0: disable
1: enable (default)
```

4. imgResult_publish (2D detection visualization publisher)
```
0: disable (default)
1: enable
```

### How to evaluate results

**CAR_MODEL = B1:**

```
rostopic echo /CamObjFrontRight
rostopic echo /CamObjFrontCenter
rostopic echo /CamObjFrontLeft
rostopic echo /CamObjRightFront
rostopic echo /CamObjRightBack
rostopic echo /CamObjLeftFront
rostopic echo /CamObjLeftBack
rostopic echo /CamObjFrontCenter
rostopic echo /CamObjBackTop
rostopic echo /CameraDetection/polygon
```

**CAR_MODEL = B1_V2:**

```
rostopic echo /cam_obj/front_bottom_60
rostopic echo /cam_obj/front_top_far_30
rostopic echo /cam_obj/front_top_close_120
rostopic echo /cam_obj/right_front_60
rostopic echo /cam_obj/right_back_60
rostopic echo /cam_obj/left_front_60
rostopic echo /cam_obj/left_back_60
rostopic echo /cam_obj/back_top_120
rostopic echo /CameraDetection/polygon
```
