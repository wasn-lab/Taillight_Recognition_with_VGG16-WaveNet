cmake_minimum_required(VERSION 3.1)
project(image_flip)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

set(IMAGE_FLIP_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")

find_package(catkin REQUIRED COMPONENTS
    sensor_msgs
    std_msgs
    roscpp
    cv_bridge
)

catkin_package(CATKIN_DEPENDS
    sensor_msgs
    std_msgs
    roscpp
    cv_bridge
)

find_package(GFlags)
if (NOT GFLAGS_FOUND)
  message(FATAL_ERROR "libgflags not found. Please run\n apt-get install libgflags-dev")
endif ()

find_package(Glog)
if (NOT GLOG_FOUND)
  message(FATAL_ERROR "libglog not found. Please run\n apt-get install libgoogle-glog-dev")
endif()

find_package(OpenCV REQUIRED
             NO_MODULE
             PATHS /usr/local /opt/ros/kinetic
             NO_DEFAULT_PATH)

add_subdirectory(src)
