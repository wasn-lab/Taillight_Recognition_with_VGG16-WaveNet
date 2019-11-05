# Message Packege (msgs)

# Description

ROS message package contains system level message definition.
It generates messages and is used in other package.

# Message definition

## PointCloud.msg
- Header lidHeader
- PointXYZI[] pointCloud

## PointXYZI.msg
- float32 x
- float32 y
- float32 z
- uint32  intensity 

## VehInfo.msg
Header Vehheader
float32 ego_x
float32 ego_y
float32 ego_heading
float32 ego_speed
