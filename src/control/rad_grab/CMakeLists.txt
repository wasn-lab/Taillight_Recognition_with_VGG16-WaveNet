cmake_minimum_required(VERSION 2.8.3)
project(rad_grab)

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

add_executable(RadFrontSub_BBox src/RadFrontSub_BBox.cpp)
target_link_libraries(RadFrontSub_BBox ${catkin_LIBRARIES})
add_dependencies(RadFrontSub_BBox ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(RadFrontPub_txt src/RadFrontPub_txt.cpp)
target_link_libraries(RadFrontPub_txt ${catkin_LIBRARIES})
add_dependencies(RadFrontPub_txt ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


add_executable(RadFrontPub src/RadFrontPub.cpp)
target_link_libraries(RadFrontPub ${catkin_LIBRARIES})
add_dependencies(RadFrontPub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(RadFrontSub src/RadFrontSub.cpp)
target_link_libraries(RadFrontSub ${catkin_LIBRARIES})
add_dependencies(RadFrontSub ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(RadFrontSub2 src/RadFrontSub2.cpp)
target_link_libraries(RadFrontSub2 ${catkin_LIBRARIES})
add_dependencies(RadFrontSub2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(RadFrontSub_CAN src/RadFrontSub_CAN.cpp)
target_link_libraries(RadFrontSub_CAN ${catkin_LIBRARIES})
add_dependencies(RadFrontSub_CAN ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


