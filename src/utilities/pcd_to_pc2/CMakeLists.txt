cmake_minimum_required(VERSION 3.1)
project(pcd_to_pc2)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs)

find_package(GFlags)
if (NOT GFLAGS_FOUND)
  message(FATAL_ERROR "libgflags not found. Please run\n apt-get install libgflags-dev")
endif ()

# Required by Alignment: Boost and PCL.
find_package(Boost REQUIRED)
find_package(PCL REQUIRED)
# PCL Variables: see PCLConfig.cmake (usu in /usr/local/share/pcl-1.9)
# Workaround for PCL bug: see https://github.com/PointCloudLibrary/pcl/issues/776
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")

find_package(Glog)
if (NOT GLOG_FOUND)
  message(FATAL_ERROR "libglog not found. Please run\n apt-get install libgoogle-glog-dev")
endif()


catkin_package(CATKIN_DEPENDS
    sensor_msgs
    roscpp)

add_subdirectory(src)
