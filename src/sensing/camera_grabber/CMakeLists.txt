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
 CATKIN_DEPENDS
 roscpp
 std_msgs
 msgs
 image_transport
 cv_bridge
 camera_utils
)

if (NOT ${ENABLE_CAMERA_GRABBER})
  message("camera_grabber is disable in car ${CAR_MODEL}")
  return ()
endif ()

find_package(CUDA)
if (NOT ${CUDA_FOUND})
  message("CUDA not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()

## common directories
set(CAMERA_GRABBER_INCLUDE_DIRS
    ${catkin_INCLUDE_DIRS}
    ${camera_utils_SOURCE_DIR}/include
    ${car_model_BINARY_DIR}/include)


set(USE_CAMERA_GSTREAMER_GRABBER OFF)
set(USE_CAMERA_DRIVEWORKS_GRABBER OFF)
if (CAR_MODEL_IS_B1_V3 OR CAR_MODEL_IS_C1)
  set(USE_CAMERA_GSTREAMER_GRABBER ON)
else ()
  set(USE_CAMERA_DRIVEWORKS_GRABBER ON)
endif ()

if (CAR_MODEL_IS_B1_V2)
  add_executable(${PROJECT_NAME}_repub_node src/image_repub_b1_v2.cpp)
  target_link_libraries(${PROJECT_NAME}_repub_node ${catkin_LIBRARIES} camera_utils)
  add_dependencies(${PROJECT_NAME}_repub_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
endif ()

## Find OpenCV
find_package(OpenCV REQUIRED PATHS /usr/local NO_DEFAULT_PATH)

if (USE_CAMERA_GSTREAMER_GRABBER)
  find_package(GStreamer)
  find_package(GLIB2)
  if (GSTREAMER_FOUND AND GLIB2_FOUND)
    message(STATUS "Find gstreamer-1.0 and glib2")
  else ()
    message(FATAL_ERROR "GStreamer (ver 1.0) is not found. Use\n sudo apt-get install libgstreamer1.0-dev libglib2.0-dev")
  endif ()
endif ()

set(DRIVEWORKS_FOUND FALSE)
if (USE_CAMERA_DRIVEWORKS_GRABBER)
  find_library(DRIVEWORKS_LIBRARY libdriveworks.so
               HINTS /usr/local/driveworks/lib)
  if (DRIVEWORKS_LIBRARY)
    set(DRIVEWORKS_FOUND TRUE)
  endif()
  if(CPU_IS_ARM)
  # Find Grabber
    find_library(ITIR_DRIVEWORKS_GRABBER_LIBRARY grabber_core
                 HINTS ${CMAKE_CURRENT_LIST_DIR}/src/CameraGrabber)
  else()
  # X86: use grabber_core_pc make build process passed.
    find_library(ITIR_DRIVEWORKS_GRABBER_LIBRARY grabber_core_pc
                 HINTS ${CMAKE_CURRENT_LIST_DIR}/src/CameraGrabber)
  endif()
endif ()

find_package(Glog)
if (NOT GLOG_FOUND)
  message(FATAL_ERROR "glog not found. Please run\n sudo apt-get install libgoogle-glog-dev")
endif()

find_package(GFlags)
if (NOT GFLAGS_FOUND)
  message(FATAL_ERROR "libgflags not found. Please run\n apt-get install libgflags-dev")
endif ()

list(APPEND CAMERA_GRABBER_INCLUDE_DIRS
  ${OpenCV_INCLUDE_DIRS}
  ${camera_utils_SOURCE_DIR}/include
  ${GLOG_INCLUDE_DIRS}
  src/Framework
  src/Util
  src/CameraGrabber
  src/RosImagePubSub
  ${DRIVENET_INCLUDE})

if (USE_CAMERA_GSTREAMER_GRABBER)
  list(APPEND CAMERA_GRABBER_INCLUDE_DIRS
  ${GSTREAMER_INCLUDE_DIRS}
  ${GLIB2_INCLUDE_DIR})
endif ()

set(FRAMEWORK_DEPS
    ${GLOG_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    ${OpenCV_LIBS}
    camera_utils
    ${catkin_LIBRARIES})

if (USE_CAMERA_GSTREAMER_GRABBER)
    list(APPEND FRAMEWORK_DEPS ${GSTREAMER_LIBRARIES})
endif()

# set include dirs before setting up build targets
include_directories(${CAMERA_GRABBER_INCLUDE_DIRS})

set(FRAMEWORK_SOURCES
  src/Framework/CameraPerception.cpp
  src/Framework/PerceptionOutput.cpp
  src/Util/ProgramArguments.cpp
  src/Util/image_utils.cpp
  src/Util/grabber_args_parser.cpp
  src/RosImagePubSub/RosImagePubSub.cpp)
cuda_add_library(${PROJECT_NAME}_framework STATIC ${FRAMEWORK_SOURCES})
target_link_libraries(${PROJECT_NAME}_framework ${FRAMEWORK_DEPS})
add_dependencies(${PROJECT_NAME}_framework ${catkin_EXPORTED_TARGETS})

if (USE_CAMERA_GSTREAMER_GRABBER)
  set(DEMO_MODE_SOURCE src/JetsonXavierGrabber.cpp)
elseif (USE_CAMERA_DRIVEWORKS_GRABBER)
  set(DEMO_MODE_SOURCE
      src/TegraAGrabber.cpp
      src/TegraBGrabber.cpp
      src/DriveXavierGrabber.cpp)
endif()

add_library(${PROJECT_NAME}_demo OBJECT ${DEMO_MODE_SOURCE})
add_dependencies(${PROJECT_NAME}_demo ${PROJECT_NAME}_framework ${catkin_EXPORTED_TARGETS})

if (USE_CAMERA_GSTREAMER_GRABBER)
  cuda_add_executable(${PROJECT_NAME}_node src/main_jetson_xavier.cpp
                      $<TARGET_OBJECTS:${PROJECT_NAME}_demo>)
  target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME}_framework)
  add_dependencies(${PROJECT_NAME}_node ${PROJECT_NAME}_demo ${PROJECT_NAME}_framework)
endif ()

if (USE_CAMERA_DRIVEWORKS_GRABBER AND DRIVEWORKS_FOUND)
  cuda_add_executable(${PROJECT_NAME}_node src/main.cpp
                      $<TARGET_OBJECTS:${PROJECT_NAME}_demo>)
  target_link_libraries(${PROJECT_NAME}_node
                        ${PROJECT_NAME}_framework
                        ${ITIR_DRIVEWORKS_GRABBER_LIBRARY}
                        ${DRIVEWORKS_LIBRARY})
  add_dependencies(${PROJECT_NAME}_node ${PROJECT_NAME}_demo ${PROJECT_NAME}_framework)
endif ()
