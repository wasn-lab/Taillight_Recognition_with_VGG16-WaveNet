前幾層的資料夾是有包含ROS的，純車尾燈辨識的程式在 itriadv/src/sensing/ itri_vehicle_signal/model 底下

Rear Signal Dataset
http://vllab1.ucmerced.edu/~hhsu22/rear_signal/rear_signal

If just want to use pre-train model to run the VGG16-WaveNet, go to itriadv/src/sensing/ itri_vehicle_signal/model, and run clasify.py

pre-train model (不知道學校的gmail甚麼時候會關起來)
Turn light : https://drive.google.com/file/d/1LkLLo_ki_tr7zLdrQl2mT9PABS7qGdcY/view?usp=sharing 
Brake light : https://drive.google.com/file/d/1khSyjkEBZW-zH3pzMszdr8MGXV2JYVy1/view?usp=sharing

This repository contains the source codes of self-driving car maintained by ITRI ICL-U.

### Prerequisite for ROS

1. Ubuntu 18.04
1. ROS Melodic
1. OpenCV 4.2.0
1. PCL 1.9.1
1. Cuda (Optional)
1. TensorRT (Optional)

See [Automatic Scripts](https://gitlab.itriadv.co/self_driving_bus/automatic_scripts)
about how to install tem.

### How to build

```sh
catkin_make -DCMAKE_BUILD_TYPE=Release
```

Tips: Run ```sudo apt-get install ccache``` to speed up the build process.

### Developers Guide

1. [How to commit code](docs/commit_code.md)
1. [Coding style](docs/coding_style.md)
