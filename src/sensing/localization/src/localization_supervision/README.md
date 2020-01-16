# Localization Supervision module

## Prerequisites
* PCL
* Ros-velodyne
 
## Getting started
```
catkin_make
source devel/seteup.bash
roslaunch localization_supervision localization.launch
```

## Output ros msg
| topic name | data type |
| --- | --- |
| /locaization_state | std_msgs::Int32 |

## locaization_state meaning
| states |  meaning  | WARNING or FATAL |
| --- | --- | --- |
| 0 | stable | NONE |
| 1 | low_gnss_frequency | WARNING |
| 2 | low_lidar_frequency | WARNING |
| 4 | low_pose_frequency | WARNING |
| 8 | pose_unstable | FATAL |
| 3 | 1+2 | WARNING |
| 5 | 1+4 | WARNING |
| 6 | 2+4 | WARNING |
| 7 | 1+2+4 | WARNING |
| 9 | 1+8 | FATAL |
| 10 | 2+8 | FATAL |
| 11 | 1+2+8 | FATAL |
| 12 | 4+8 | FATAL |
| 13 | 1+4+8 | FATAL |
| 15 | 1+2+4+8 | FATAL |

