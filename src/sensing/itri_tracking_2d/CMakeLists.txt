cmake_minimum_required(VERSION 3.1)
project(itri_tracking_2d)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  std_msgs
  msgs
  pcl_ros
  grid_map_ros
  hungarian
)

find_package(Glog)

find_package(GFlags)
if (NOT GFLAGS_FOUND)
  message(FATAL_ERROR "libgflags not found. Please run\n apt-get install libgflags-dev")
endif ()

find_package(Eigen3 REQUIRED)

find_package(OpenCV REQUIRED
  NO_MODULE
  PATHS /usr/local /opt/ros/kinetic
  NO_DEFAULT_PATH
)

catkin_package(CATKIN_DEPENDS)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIR}
  ${hungarian_SOURCE_DIR}/src     
)

set(SOURCES_SRCS
  src/kalman_tracker.cpp
  src/kalman_trackers.cpp
  src/main.cpp
  src/track2d_node.cpp
  src/track_hist.cpp
  src/utils.cpp
  src/velocity.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES_SRCS})

if(EXISTS /usr/local/lib/libopencv_core.so)
target_link_libraries(
  ${PROJECT_NAME}
  /usr/local/lib/libopencv_core.so
  ${catkin_LIBRARIES}
  glog
  gflags
  ${OpenCV_LIBRARIES}
  hungarian
)
else()
target_link_libraries(
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  glog
  gflags
  ${OpenCV_LIBRARIES}
  hungarian
)
endif()

add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
