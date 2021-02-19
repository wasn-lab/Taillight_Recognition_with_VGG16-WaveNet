cmake_minimum_required(VERSION 3.1)
project(drivable_area_filter)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf2_geometry_msgs
  std_msgs
  nav_msgs
  msgs
)

find_package(Glog)

find_package(GFlags)
if (NOT GFLAGS_FOUND)
  message(FATAL_ERROR "libgflags not found. Please run\n apt-get install libgflags-dev")
endif ()

find_package(OpenCV REQUIRED
  NO_MODULE
  PATHS /usr/local /opt/ros/kinetic
  NO_DEFAULT_PATH
)

catkin_package(CATKIN_DEPENDS)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${msgs_SOURCE_DIR}/include
)

set(SOURCES_SRCS
  src/main.cpp
  src/daf_node.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES_SRCS})

target_link_libraries(
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  glog
  gflags
  ${OpenCV_LIBRARIES}
)

add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
