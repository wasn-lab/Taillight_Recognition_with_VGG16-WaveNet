cmake_minimum_required(VERSION 2.8.3)
project(system_time)
 
set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})
include(CompilerFlags)
include(CarModel)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
)  

catkin_package(
  CATKIN_DEPENDS roscpp rospy msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

cuda_add_executable(${PROJECT_NAME}_ros_time src/ros_time.cpp)
target_link_libraries(${PROJECT_NAME}_ros_time ${catkin_LIBRARIES})
add_dependencies(${PROJECT_NAME}_ros_time ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
