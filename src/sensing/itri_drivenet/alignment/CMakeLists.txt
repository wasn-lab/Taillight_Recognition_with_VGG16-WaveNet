cmake_minimum_required(VERSION 2.8.3)
project(alignment)
 
set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  cv_bridge
  msgs
  car_model
  camera_utils
)  

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp rospy msgs
)

######################
##      OPENCV      ##
######################
find_package(OpenCV REQUIRED PATHS /usr/local NO_DEFAULT_PATH)

######################
##       CUDA       ##
######################
find_package	        (CUDA REQUIRED) 
if (NOT ${CUDA_FOUND})
  message("CUDA not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()

######################
##       PCL        ##
######################
find_package            (PCL 1.8 REQUIRED) 
include_directories     (${PCL_INCLUDE_DIRS})
link_directories        (${PCL_LIBRARY_DIRS})
add_definitions         (${PCL_DEFINITIONS})
include_directories     (${CUDA_INCLUDE_DIRS}) 
message("CUDA found: ${CUDA_FOUND} CUDA version: ${CUDA_VERSION_STRING}")

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${car_model_BINARY_DIR}/include
  ${camera_utils_SOURCE_DIR}/include
)

set(LIBS ${OpenCV_LIBRARIES}
         ${CUDA_LIBRARIES}
         ${PCL_LIBRARIES}
         camera_utils)

file(GLOB_RECURSE LIB_SOURCES *.cpp)
message(STATUS "LIB_SOURCES: ${LIB_SOURCES}")
cuda_add_executable(${PROJECT_NAME}_b1 src/main_b1.cpp src/alignment.cpp)
target_link_libraries(${PROJECT_NAME}_b1 ${catkin_LIBRARIES} ${LIBS})
add_dependencies(${PROJECT_NAME}_b1 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
