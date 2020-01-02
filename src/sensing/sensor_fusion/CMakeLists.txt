cmake_minimum_required(VERSION 3.1)
project(sensor_fusion)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 11)  # -std=c++11
set(CMAKE_C_STANDARD 11)  # -std=gnu11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-parameter")

set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  msgs
)

find_package(PCL 1.9.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED
             NO_MODULE
             PATHS /usr/local /opt/ros/kinetic
             NO_DEFAULT_PATH)

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

catkin_package(
  LIBRARIES ${PROJECT_NAME}
  roscpp 
  rospy 
  sensor_msgs
  std_msgs
  msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${msgs_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(${PROJECT_NAME} src/sensor_fusion.cpp src/ROSPublish.cpp)
 
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

add_dependencies(${PROJECT_NAME}
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
