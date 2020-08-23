cmake_minimum_required(VERSION 2.8.3)
project(rad_grab)

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

find_package(PCL)


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

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

###########
## Build ##
###########
include_directories(${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

add_executable(radfront_delphi src/RadFrontDelphi.cpp)
target_link_libraries(radfront_delphi ${catkin_LIBRARIES})
add_dependencies(radfront_delphi ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(radfront_alpha src/RadFrontAlpha.cpp)
target_link_libraries(radfront_alpha ${catkin_LIBRARIES})
add_dependencies(radfront_alpha ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(radar_all src/RadarAll.cpp)
target_link_libraries(radar_all ${catkin_LIBRARIES})
add_dependencies(radar_all ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

#add_executable(radfrontsub_rviz src/RadFrontSub_rviz.cpp)
#target_link_libraries(radfrontsub_rviz ${catkin_LIBRARIES})
#add_dependencies(radfrontsub_rviz ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(radfrontsub_can src/RadFrontSub_CAN.cpp)
target_link_libraries(radfrontsub_can ${catkin_LIBRARIES})
add_dependencies(radfrontsub_can ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(radfrontsub_pcloud src/RadFrontSub_PCloud.cpp)
target_link_libraries(radfrontsub_pcloud ${catkin_LIBRARIES})
add_dependencies(radfrontsub_pcloud ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(radfrontsub_bbox src/RadFrontSub_BBox.cpp)
target_link_libraries(radfrontsub_bbox ${catkin_LIBRARIES})
add_dependencies(radfrontsub_bbox ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
