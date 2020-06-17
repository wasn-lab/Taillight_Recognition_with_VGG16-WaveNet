This repository contains the source codes of self-driving car maintained by ITRI ICL-U.

### Prerequisite

1. Ubuntu 18.04
1. ROS Melodic
1. OpenCV 4.2.0 (Optional)
1. PCL 1.9.1 (Optional)
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
