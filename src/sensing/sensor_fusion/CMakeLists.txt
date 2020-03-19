cmake_minimum_required(VERSION 3.1)
project(sensor_fusion)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(
  catkin REQUIRED COMPONENTS
  roscpp
  msgs
  hungarian
)

catkin_package(
  LIBRARIES
  ${PROJECT_NAME}
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${msgs_INCLUDE_DIRS}
  ${hungarian_SOURCE_DIR}/src
)

add_executable(
  ${PROJECT_NAME}
  src/sensor_fusion.cpp
)
 
target_link_libraries(
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  hungarian
)

add_dependencies(
  ${PROJECT_NAME}
  ${catkin_EXPORTED_TARGETS}
)
