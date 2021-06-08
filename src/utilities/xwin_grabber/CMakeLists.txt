cmake_minimum_required(VERSION 3.1)
project(xwin_grabber)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

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

find_package(X11)
if (NOT X11_FOUND)
  message("Cannot find X11 libraries. Skip building this package")
  return()
endif ()

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
