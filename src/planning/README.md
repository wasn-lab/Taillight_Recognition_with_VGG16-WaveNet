### Install required ROS packages for lanelet2_extension package.
```
sudo apt install ros-melodic-mrt-cmake-modules ros-melodic-lanelet2 ros-melodic-lanelet2-core ros-melodic-lanelet2-io ros-melodic-lanelet2-maps ros-melodic-lanelet2-projection ros-melodic-lanelet2-routing ros-melodic-lanelet2-traffic-rules ros-melodic-lanelet2-validation
```

### Install required non-ROS packages for for lanelet2_extension package.
```
sudo apt install libpugixml-dev libgeographic-dev 
```

### Install required ROS packages for map_based_prediction package.
```
sudo apt install ros-melodic-uuid-msgs ros-melodic-unique-id
```

### Install vehicle description
```
sudo apt-get install ros-melodic-automotive-platform-msgs ros-melodic-automotive-navigation-msgs ros-melodic-pacmod-msgs
``` 

### Install osqp
1. Clone this repository
```
git clone https://github.com/tier4/AutowareArchitectureProposal.git
cd AutowareArchitectureProposal/
```
2. Run the setup script
```
./setup_ubuntu18.04.sh
```
Note : Do you install GPU modules(cuda: 10.2, cudnn: 7.6.5, TensorRT: 7.0.0)? (y/n) -> choose "n"
