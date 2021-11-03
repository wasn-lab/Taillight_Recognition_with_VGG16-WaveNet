### lanelet2_extension

此為lanelet2所需之package,不過由於現在使用ros-melodic,所以此package應該是不用了,不過先保留以防萬一

### map loader

主要load map的pkg,目前只使用到裡面的lanelet2_map_loader,詳細可看map_loader/README.md


### map_tf_generator

主要生成map與viewer中間的frame tf,viewer主要用於planning_simulator

### util

目前沒有特別研究此pkg


### Install packages

* Install required ROS packages for lanelet2_extension package.

```
sudo apt install ros-melodic-mrt-cmake-modules ros-melodic-lanelet2 ros-melodic-lanelet2-core ros-melodic-lanelet2-io ros-melodic-lanelet2-maps ros-melodic-lanelet2-projection ros-melodic-lanelet2-routing ros-melodic-lanelet2-traffic-rules ros-melodic-lanelet2-validation
```

* Install required non-ROS packages for for lanelet2_extension package.

```
sudo apt install libpugixml-dev libgeographic-dev 
```

* Install required ROS packages for map_based_prediction package.

```
sudo apt install ros-melodic-uuid-msgs ros-melodic-unique-id
```
