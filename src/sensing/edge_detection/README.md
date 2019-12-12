# Edge detection module
## Prerequisites
* PCL
* ros-kinetic-velodyne
* ros-kinetic-grid-map

## Getting started
```
catkin_make
source devel/seteup.bash
roslaunch edge_detection edge_detection.launch
```

## Input and Output 
| Input topic name |  data type  |
| --- | --- |
| /LidarFrontTop  | sensor_msgs::PointCloud2 |
| /LidarFrontRight | sensor_msgs::PointCloud2 |
| /LidarFrontLeft| sensor_msgs::PointCloud2 |

| Output | data type |
| --- | --- |
| /occupancy_grid | nav_msgs::OccupancyGrid |
| /ring_edge_point_cloud | sensor_msgs::PointCloud2 |
