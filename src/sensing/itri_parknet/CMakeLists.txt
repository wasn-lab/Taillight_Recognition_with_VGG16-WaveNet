cmake_minimum_required(VERSION 3.1)
project(itri_parknet)

find_package(catkin REQUIRED COMPONENTS
    cv_bridge
    image_transport
    sensor_msgs
    std_msgs
    message_generation
    msgs
    car_model
)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})
include(CarModel)

if (NOT ENABLE_PARKNET)
  message(STATUS "${PROJECT_NAME} is disabled in car ${CAR_MODEL}")
  return ()
endif ()

include(CompilerFlags)

set(CAMERA_TEGRAB_DIR ${Project_SOURCE_DIR}/sensing/camera_tegrab)
option(USE_YOLO_V3 "Use yolo v3" ON)
option(USE_CUDA "Use Cuda to seepd up" ON)
option(USE_TENSORRT "Use TensorRT to speed up" ON)
option(ENABLE_GCOV "Use gcov tool to see code coverage" OFF)

set(PARKNET_CFG_DIR "${CMAKE_CURRENT_LIST_DIR}/cfg")
set(PARKNET_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/src")
set(PARKNET_WEIGHTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/weights)


find_package(TensorRT)
if (USE_TENSORRT AND NOT TensorRT_FOUND)
  message("Cannot find TensorRT. Skip ${PROJECT_NAME}")
  return()
endif()

#set(OpenCV_DIR "/opt/ros/kinetic/share/OpenCV-3.3.1-dev" CACHE STRING "")
find_package(OpenCV REQUIRED
             NO_MODULE
             PATHS /usr/local /opt/ros/kinetic
             NO_DEFAULT_PATH)
find_path(OpenCV_CUDAIMGPROC_LIB
  libopencv_cudaimgproc.so
  PATHS /usr/local/lib
)
if (NOT OpenCV_CUDAIMGPROC_LIB)
  set(USE_OPENCV_CUDA OFF)
else ()
  set(USE_OPENCV_CUDA ON)
endif ()

find_package(CUDA)

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

# For code coverage analysis
if (ENABLE_GCOV AND CMAKE_COMPILER_IS_GNUCXX)
  include(cmake/CodeCoverage.cmake)
  APPEND_COVERAGE_COMPILER_FLAGS()
  SETUP_TARGET_FOR_COVERAGE_LCOV(
    NAME coverage
    EXECUTABLE ${CATKIN_DEVEL_PREFIX}/lib/pslot_detector/pslot_detector -score_thresh 0.05 -nms_thresh 0.1 -display_gui=false)
endif ()

if (USE_CUDA AND NOT CUDA_FOUND)
  message("Turn off USE_CUDA since Cuda is not installed.")
  set(USE_CUDA OFF)
endif ()

if (USE_TENSORRT)
  if (NOT CUDA_FOUND)
    message("Turn off USE_TENSORRT since USE_CUDA is FALSE.")
    set(USE_TENSORRT OFF)
  endif()
endif ()

if (USE_TENSORRT)
  set(USE_DARKNET OFF)
else ()
  set(USE_DARKNET ON)
endif ()

set(PARKNET_CATKIN_DEPS
    cv_bridge
    image_transport
    sensor_msgs
    message_runtime
    msgs
    camera_utils
    std_msgs)

catkin_package(CATKIN_DEPENDS ${PARKNET_CATKIN_DEPS})

set(PARKNET_TEST_DATA_DIR "${PARKNET_SRC_DIR}/test_data")
include(cmake/yolo_files.cmake)
configure_file(include/parknet.h.in
               ${PROJECT_BINARY_DIR}/include/parknet.h)

if (USE_DARKNET)
  add_subdirectory(darknet)
endif ()
add_subdirectory(yolo_tensorrt)
add_subdirectory(src)
#include(cmake/dl_weights.cmake)
