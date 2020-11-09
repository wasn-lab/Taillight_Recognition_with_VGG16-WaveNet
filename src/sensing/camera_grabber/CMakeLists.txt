cmake_minimum_required(VERSION 2.8.3)
project(camera_grabber)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})
include(CompilerFlags)
include(CarModel)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  msgs
  image_transport
  cv_bridge
  car_model
  camera_utils
)

catkin_package(
#  INCLUDE_DIRS include
#   LIBRARIES
   CATKIN_DEPENDS
   roscpp
   std_msgs
   msgs
   image_transport
   cv_bridge
   camera_utils
#  DEPENDS system_lib
)

## common directories
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${camera_utils_SOURCE_DIR}/include
  ${car_model_BINARY_DIR}/include)

## 1. create repub image node
if (CAR_MODEL_IS_B1_V2)
  add_executable(
      ${PROJECT_NAME}_repub_node src/image_repub_b1_v2.cpp)
  target_link_libraries(${PROJECT_NAME}_repub_node ${catkin_LIBRARIES} camera_utils)
  add_dependencies(${PROJECT_NAME}_repub_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
endif ()

## 2. create camera grabber node
if (NOT ${ENABLE_CAMERA_GRABBER})
  message("camera_grabber is disable in car ${CAR_MODEL}")
  return ()
endif ()

## Find CUDA
find_package(CUDA)
if (NOT ${CUDA_FOUND})
  message("CUDA not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()

if (COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-attribute=format")
endif ()

## Find OpenCV
find_package(OpenCV REQUIRED PATHS /usr/local NO_DEFAULT_PATH)


if (NOT ${ENABLE_CAMERA_GSTREAMER_GRABBER})
## Find Driveworks 2.0
find_library(DRIVEWORKS_LIBRARY driveworks
  HINTS /usr/local/driveworks-2.0/ /usr/local/driveworks/ PATH_SUFFIXES lib lib/x64)
else ()
## Find Gstreamer
find_package(PkgConfig) #finding pkg-config is a helper tool
#using pkg-config to getting Gstreamer
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
endif ()


if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  find_library(DRIVEWORKS_LIBRARY libdriveworks.so
      HINTS /usr/local/driveworks/lib)
  if (NOT DRIVEWORKS_LIBRARY)
    return()
  endif()
endif()

## Find Grabber
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  find_library(GRABBER_LIBRARY grabber_core
               HINTS ${CMAKE_CURRENT_LIST_DIR}/src/CameraGrabber)
else()
  find_library(GRABBER_LIBRARY grabber_core_pc
               HINTS ${CMAKE_CURRENT_LIST_DIR}/src/CameraGrabber)
endif()

## Find Glog
find_package(Glog)
if (NOT GLOG_FOUND)
  message(FATAL_ERROR "glog not found. Please run\n sudo apt-get install libgoogle-glog-dev")
endif()

## Find GFlags
find_package(GFlags)
if (NOT GFLAGS_FOUND)
  message(FATAL_ERROR "libgflags not found. Please run\n apt-get install libgflags-dev")
endif ()


if (NOT ${ENABLE_CAMERA_GSTREAMER_GRABBER})
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${car_model_BINARY_DIR}/include
  ${camera_utils_SOURCE_DIR}/include
  ${GLOG_INCLUDE_DIRS}
  src/Framework
  src/Util
  src/CameraGrabber
  src/RosImagePubSub
  ${DRIVENET_INCLUDE}  
)
else ()
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${car_model_BINARY_DIR}/include
  ${camera_utils_SOURCE_DIR}/include
  ${GLOG_INCLUDE_DIRS}
  src/Framework
  src/Util
  src/CameraGrabber
  src/RosImagePubSub
  ${DRIVENET_INCLUDE}
  ${GSTREAMER_INCLUDE_DIRS}
)
endif ()

set(FRAMEWORK_SOURCES
  src/Framework/CameraPerception.cpp
  src/Framework/PerceptionOutput.cpp
  src/Util/ProgramArguments.cpp
  src/Util/image_utils.cpp
  src/Util/grabber_args_parser.cpp
  src/RosImagePubSub/RosImagePubSub.cpp)

if (NOT ${ENABLE_CAMERA_GSTREAMER_GRABBER})
set(FRAMEWORK_DEPS
    ${GLOG_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    ${OpenCV_LIBS}
    camera_utils
    ${catkin_LIBRARIES})
else ()
set(FRAMEWORK_DEPS
    ${GLOG_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    ${OpenCV_LIBS}
    camera_utils
    ${catkin_LIBRARIES}
    ${GSTREAMER_LIBRARIES})
endif()

cuda_add_library(${PROJECT_NAME}_framework STATIC ${FRAMEWORK_SOURCES})
target_link_libraries(${PROJECT_NAME}_framework ${FRAMEWORK_DEPS})
add_dependencies(${PROJECT_NAME}_framework ${catkin_EXPORTED_TARGETS})

if (NOT ${ENABLE_CAMERA_GSTREAMER_GRABBER})
set(DEMO_MODE_SOURCE
  src/TegraAGrabber.cpp
  src/TegraBGrabber.cpp
  src/DriveXavierGrabber.cpp)
else ()
set(DEMO_MODE_SOURCE
  src/JetsonXavierGrabber.cpp)
endif()

add_library(${PROJECT_NAME}_demo OBJECT ${DEMO_MODE_SOURCE})
add_dependencies(${PROJECT_NAME}_demo ${PROJECT_NAME}_framework ${catkin_EXPORTED_TARGETS})

if (NOT ${ENABLE_CAMERA_GSTREAMER_GRABBER})
 if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  cuda_add_executable(
      ${PROJECT_NAME}_node src/main.cpp
      $<TARGET_OBJECTS:${PROJECT_NAME}_demo>)
  target_link_libraries(${PROJECT_NAME}_node
      ${PROJECT_NAME}_framework
      ${GRABBER_LIBRARY}
      ${DRIVEWORKS_LIBRARY})
  add_dependencies(${PROJECT_NAME}_node ${PROJECT_NAME}_demo ${PROJECT_NAME}_framework)
 endif ()
else()
 if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  cuda_add_executable(
      ${PROJECT_NAME}_node src/main_jetson_xavier.cpp
      $<TARGET_OBJECTS:${PROJECT_NAME}_demo>)
  target_link_libraries(${PROJECT_NAME}_node
      ${PROJECT_NAME}_framework      
      ${DRIVEWORKS_LIBRARY})
  add_dependencies(${PROJECT_NAME}_node ${PROJECT_NAME}_demo ${PROJECT_NAME}_framework)
 endif ()
endif ()
