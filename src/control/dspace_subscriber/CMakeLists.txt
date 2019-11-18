cmake_minimum_required(VERSION 2.8.3)
project(dspace_subscriber)


## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

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
generate_messages(
  DEPENDENCIES
  std_msgs
  msgs
)

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


add_executable(dspace_subscriber src/dSPACE_subscriber.cpp )
target_link_libraries(dspace_subscriber ${catkin_LIBRARIES})
add_dependencies(dspace_subscriber dspace_subscriber_generate_messages_cpp)
add_dependencies(dspace_subscriber ${${PROJECT_NAME}_EXPORTED_TARGETS})







