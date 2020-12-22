cmake_minimum_required(VERSION 3.1)
project(deeplab)

set(CATKIN_REQUIRED_COMPONENTS
  camera_utils
  car_model
  cv_bridge
  image_transport
  roscpp
  roslib
  sensor_msgs
)
find_package(catkin REQUIRED COMPONENTS ${CATKIN_REQUIRED_COMPONENTS})

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})
include(CarModel)

include(CompilerFlags)

catkin_package(CATKIN_DEPENDS ${CATKIN_REQUIRED_COMPONENTS})

find_package(OpenCV REQUIRED)
find_package(GFlags)
if (NOT GFLAGS_FOUND)
  message(FATAL_ERROR "libgflags not found. Please run\n apt-get install libgflags-dev")
endif ()

find_package(Glog)
if (NOT GLOG_FOUND)
  message(FATAL_ERROR "libglog not found. Please run\n apt-get install libgoogle-glog-dev")
endif()

add_subdirectory(src)
