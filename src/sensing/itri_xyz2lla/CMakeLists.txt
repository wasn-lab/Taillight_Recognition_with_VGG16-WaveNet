cmake_minimum_required(VERSION 3.1)
project(itri_xyz2lla)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(
  catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  tf2_geometry_msgs
  msgs
  gnss_utility
  tf2_ros
)

catkin_package(CATKIN_DEPENDS)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${msgs_SOURCE_DIR}/include
)

set(SOURCES_SRCS
  src/main.cpp
  src/xyz2lla.cpp
)

add_executable(
  ${PROJECT_NAME}
  ${SOURCES_SRCS}
)

target_link_libraries(
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

add_dependencies(
  ${PROJECT_NAME}
  ${catkin_EXPORTED_TARGETS}
)
