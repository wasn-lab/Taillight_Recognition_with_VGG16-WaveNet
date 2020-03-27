cmake_minimum_required(VERSION 2.8.3)
project(geofence_pp)


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
find_package(PCL)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

###########
## Build ##
###########
include_directories(${catkin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
)


##add_executable(Geofence_old src/ert_main.cpp src/Geofence.cpp )
##target_link_libraries(Geofence_old ${catkin_LIBRARIES} ${PCL_LIBRARIES})
##add_dependencies(Geofence_old ${catkin_EXPORTED_TARGETS} )


add_executable(geofence_pp src/main.cpp src/Geofence_Class.cpp )
target_link_libraries(geofence_pp ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(geofence_pp ${catkin_EXPORTED_TARGETS} )
#add_dependencies(geofence_pp ${${PROJECT_NAME}_EXPORTED_TARGETS})
