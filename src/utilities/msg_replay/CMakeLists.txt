cmake_minimum_required(VERSION 3.1)
project(msg_replay)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(catkin REQUIRED COMPONENTS
    sensor_msgs
    std_msgs
    roscpp
    msgs
)

catkin_package(CATKIN_DEPENDS
    sensor_msgs
    std_msgs
    roscpp
    msgs
)

find_package(GFlags)
if (NOT GFLAGS_FOUND)
  message(FATAL_ERROR "libgflags not found. Please run\n apt-get install libgflags-dev")
endif ()

find_package(Glog)
if (NOT GLOG_FOUND)
  message(FATAL_ERROR "libglog not found. Please run\n apt-get install libgoogle-glog-dev")
endif()

add_subdirectory(src)
