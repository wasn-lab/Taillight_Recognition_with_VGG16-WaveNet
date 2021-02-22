cmake_minimum_required(VERSION 3.1)
project(pc2_compressor)

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

# Snappy is a lossless compression algorithm.
# It is faster than lfz but has lower compression ratio
#
find_package(Snappy)
if (NOT Snappy_FOUND)
  message(FATAL_ERROR "libsnappy not found. Please run\n sudo apt-get install libsnappy-dev")
endif()

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)

find_package(PCL 1.9.1 REQUIRED)
# PCL Variables: see PCLConfig.cmake (usu in /usr/local/share/pcl-1.9)
# Workaround for PCL bug: see https://github.com/PointCloudLibrary/pcl/issues/776
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")

add_subdirectory(src)
