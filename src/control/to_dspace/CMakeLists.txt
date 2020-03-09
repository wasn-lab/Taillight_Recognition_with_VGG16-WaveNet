cmake_minimum_required(VERSION 2.8.3)
project(to_dspace)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  msgs
)

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
)

###########
## Build ##
###########
include_directories(${catkin_INCLUDE_DIRS})


add_executable(to_dspace src/to_dspace.cpp )
target_link_libraries(to_dspace ${catkin_LIBRARIES})
add_dependencies(to_dspace ${catkin_EXPORTED_TARGETS})

add_executable(bus_stop_info src/bus_stop_info_new.cpp )
target_link_libraries(bus_stop_info ${catkin_LIBRARIES})
add_dependencies(bus_stop_info ${catkin_EXPORTED_TARGETS})







