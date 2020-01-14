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

## Output of rostopic /locaization_states 
| states |  meaning  |   
| --- | --- |
| 1 | state_gnss_delay |
| 2 | state_lidar_delay |
| 4 | state_pose_delay |
| 8 | pose_unstable |
| 3 | 1+2 |
| 5 | 1+4 |
| 6 | 2+4 |
| 7 | 1+2+4 |
| 9 | 1+8 |
| 10 | 2+8 |
| 11 | 1+2+8 |
| 12 | 4+8 |
| 13 | 1+4+8 |
| 15 | 1+2+4+8 |

