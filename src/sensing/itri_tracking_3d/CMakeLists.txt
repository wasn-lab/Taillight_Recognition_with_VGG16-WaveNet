cmake_minimum_required(VERSION 3.1)
project(itri_tracking_3d)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf2_geometry_msgs
  std_msgs
  msgs
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
  src/ego_param.cpp
  src/kalman_tracker.cpp
  src/kalman_trackers.cpp
  src/main.cpp
  src/marker_gen.cpp
  src/point32_impl.cpp
  src/ros_params_parser.cpp
  src/simple_kalman.cpp
  src/tpp_args_parser.cpp
  src/tpp_node.cpp
  src/track_hist.cpp
  src/utils.cpp
  src/velocity.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES_SRCS})

target_link_libraries(
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  glog
  gflags
  ${OpenCV_LIBRARIES}
  hungarian
)

add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
