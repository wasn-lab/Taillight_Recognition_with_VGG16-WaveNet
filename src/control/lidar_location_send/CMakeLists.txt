cmake_minimum_required(VERSION 2.8.3)
project(lidar_location_send)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})
include(CarModel)
include(CompilerFlags)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_generation
  genmsg
  roslib
  sensor_msgs
  msgs
  car_model
)
## Generate messages in the 'msg' folder
add_message_files(DIRECTORY msg
  FILES
  MM_TP_msg.msg
  UKF_MM_msg.msg
)
## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs msgs
)



## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
 include
  ${catkin_INCLUDE_DIRS}
)


#include_directories(include ${catkin_INCLUDE_DIRS})


add_executable(${PROJECT_NAME} src/lidar_location_send.cpp src/localization_can_class.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})
MESSAGE("Generate ${PROJECT_NAME}")
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_generate_messages_cpp )
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS} )
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS})
