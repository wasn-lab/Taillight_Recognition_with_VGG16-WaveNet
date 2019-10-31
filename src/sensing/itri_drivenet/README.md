### Install drivenet dependencies

* Install Cuda 10.0
* Install TensorRT 5
* Install PCL 1.9.1

* Install ros-kinetic-cv-bridge, ros-kinetic-opencv3 and image-transport
```
sudo apt-get install ros-kinetic-image-transport
sudo apt-get install ros-kinetic-cv-bridge 
```

### How to run

car 1:

```
source ./devel/setup.bash
```

FOV60 CamObjFrontRight, CamObjFrontCenter, CamObjFrontLeft
```
roslaunch drivenet b1_drivenet60.launch
```

FOV120 CamObjRightFront, CamObjRightBack, CamObjLeftFront, CamObjLeftBack
```
roslaunch drivenet b1_drivenet120_1.launch 
```

FOV120 CamObjFrontCenter, CamObjBackTop
```
roslaunch drivenet b1_drivenet120_2.launch
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
```

