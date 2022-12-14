cmake_minimum_required(VERSION 3.1)
project(pc_transform)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(catkin REQUIRED COMPONENTS
    sensor_msgs
    std_msgs
    roscpp
    msgs
    car_model
)

catkin_package(CATKIN_DEPENDS
    sensor_msgs
    std_msgs
    roscpp
    msgs
    car_model
)

find_package(CUDA)
if (NOT ${CUDA_FOUND})
  message("CUDA not found. Skip building ${PROJECT_NAME}")
endif ()

find_package(GFlags)
if (NOT GFLAGS_FOUND)
  message(FATAL_ERROR "libgflags not found. Please run\n apt-get install libgflags-dev")
endif ()

find_package(Glog)
if (NOT GLOG_FOUND)
  message(FATAL_ERROR "libglog not found. Please run\n apt-get install libgoogle-glog-dev")
endif()

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

find_package(PCL 1.9.1 REQUIRED)
# PCL Variables: see PCLConfig.cmake (usu in /usr/local/share/pcl-1.9)
# Workaround for PCL bug: see https://github.com/PointCloudLibrary/pcl/issues/776
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")

add_subdirectory(src)
