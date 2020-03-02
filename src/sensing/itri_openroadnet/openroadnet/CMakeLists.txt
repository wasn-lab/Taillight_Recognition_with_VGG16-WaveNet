cmake_minimum_required(VERSION 2.8.3)
project(openroadnet)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(OpenCV REQUIRED PATHS /usr/local NO_DEFAULT_PATH)
find_package(CUDA)
if (NOT ${CUDA_FOUND})
  message("CUDA not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  msgs
  libs_opn
  image_transport
  message_generation
  cv_bridge
)

include_directories     (${CUDA_INCLUDE_DIRS})
set(CUDA_LIBS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppisu.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppig.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppc.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppidei.so
                ${CUDA_cublas_LIBRARY}
                ${CUDA_curand_LIBRARY})
message("CUDA found: ${CUDA_FOUND} CUDA version: ${CUDA_VERSION_STRING}")


catkin_package(
   INCLUDE_DIRS src
#  LIBRARIES openroadnet
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
   CATKIN_DEPENDS msgs libs_opn
)

###########
## Build ##
###########
set(OpenCV_INCLUDE_DIRS "/usr/local/include")
set(OpenCV_LIBRARY_DIRS "/usr/local/lib")
set(OpenCV_LIBRARIES opencv_core
                     opencv_imgproc
                     opencv_imgcodecs
                     opencv_highgui
                     opencv_dnn
                     opencv_videoio)
set(TENSORFLOW_ROOT_DIR ${Project_SOURCE_DIR}/dl_data/assets)
set(TENSORFLOW_INCLUDE_DIRS
  ${TENSORFLOW_ROOT_DIR}/include
  ${TENSORFLOW_ROOT_DIR}/include/tensorflow/c)
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_INCLUDE}
  ${TENSORFLOW_INCLUDE_DIRS}
)

set(PLUGIN_SOURCES
src/openroadnet.cpp
src/DistanceEstimation.cpp
src/main.cpp
)

add_executable(openroadnet src/openroadnet.cpp src/DistanceEstimation.cpp src/main.cpp)
add_dependencies(openroadnet ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(openroadnet libs_opn ${catkin_LIBRARIES} ${CUDA_LIBRARIES} cuda ${CUDA_LIBS})

add_library(openroadnet_lib src/openroadnet.cpp src/DistanceEstimation.cpp)
add_dependencies(openroadnet_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(openroadnet_lib libs_opn ${catkin_LIBRARIES})
