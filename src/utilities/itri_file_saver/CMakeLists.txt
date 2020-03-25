cmake_minimum_required(VERSION 3.1)
project(itri_file_saver)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(catkin REQUIRED COMPONENTS
    cv_bridge
    image_transport
    sensor_msgs
)

find_package(OpenCV REQUIRED)
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

set(FILE_SAVER_CATKIN_DEPS
    cv_bridge
    image_transport
    sensor_msgs
    message_runtime
    std_msgs)

catkin_package(CATKIN_DEPENDS ${FILE_SAVER_CATKIN_DEPS})

add_subdirectory(src)
