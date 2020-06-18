cmake_minimum_required(VERSION 2.8.3)
project(control_checker)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  msgs
  car_model
)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})
include(CarModel)
include(CompilerFlags)

################################################
## Declare ROS messages, services and actions ##
################################################
## Generate added messages and services with any dependencies listed here
#generate_messages(
#  DEPENDENCIES
#  std_msgs
#  msgs
#)

###################################
## catkin specific configuration ##
###################################
catkin_package(CATKIN_DEPENDS
  roscpp 
  rospy 
  std_msgs 
  message_runtime 
  msgs
  car_model
)

###########
## Build ##
###########
include_directories(${catkin_INCLUDE_DIRS})


add_executable(control_checker src/control_checker.cpp )
target_link_libraries(control_checker ${catkin_LIBRARIES})
add_dependencies(control_checker ${catkin_EXPORTED_TARGETS})







