### Install drivenet dependencies
* Install ros-kinetic-cv-bridge, ros-kinetic-opencv3 and image-transport
```
sudo apt-get install ros-kinetic-image-transport
sudo apt-get install ros-kinetic-cv-bridge
```

### Compile

**CAR_MODEL = B1_V3:**
```
catkin_make -DCMAKE_BUILD_TYPE=Release -DCAR_MODEL=B1_V3
```

**CAR_MODEL = C1:**
```
catkin_make -DCMAKE_BUILD_TYPE=Release -DCAR_MODEL=C1
```

### How to run

**CAR_MODEL = B1_V3 & C1:**
```
./devel//lib/camera_calibration/camera_calibration_node
```
