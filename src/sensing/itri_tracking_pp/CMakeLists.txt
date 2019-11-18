cmake_minimum_required(VERSION 3.1)
project(itri_tracking_pp)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 11)  # -std=c++11
set(CMAKE_C_STANDARD 11)  # -std=gnu11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-parameter")

set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")

find_package(catkin REQUIRED COMPONENTS
        message_generation
        roscpp
        geometry_msgs
        tf2_geometry_msgs
        sensor_msgs
        std_msgs
        msgs
)

find_package(GFlags)
if (NOT GFLAGS_FOUND)
  message(FATAL_ERROR "libgflags not found. Please run\n apt-get install libgflags-dev")
endif ()

find_package(Glog)

find_package(OpenCV 3 REQUIRED
             NO_MODULE
             PATHS /usr/local /opt/ros/kinetic
             NO_DEFAULT_PATH)

find_package(Eigen3 REQUIRED)

set(TPP_CATKIN_DEPS
    message_runtime
    roscpp
    geometry_msgs
    tf2_geometry_msgs
    sensor_msgs
    std_msgs
    msgs
)

catkin_package(CATKIN_DEPENDS ${TPP_CATKIN_DEPS})

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIR}
)

set(SOURCES_SRCS
    src/ar1.cpp
    src/ego_param.cpp
    src/hungarian.cpp
    src/kalman_tracker.cpp
    src/kalman_trackers.cpp
    src/main.cpp
    src/marker_gen.cpp
    src/path_predict.cpp
    src/point32_impl.cpp
    src/ros_params_parser.cpp
    src/simple_kalman.cpp
    src/tpp_args_parser.cpp
    src/tpp_node.cpp
    src/track_hist.cpp
    src/utils.cpp
    src/velocity.cpp
)

add_executable(itri_tracking_pp ${SOURCES_SRCS})

target_link_libraries(
        itri_tracking_pp
        ${catkin_LIBRARIES}
        glog
        gflags
        ${OpenCV_LIBRARIES}
)

add_dependencies(itri_tracking_pp ${catkin_EXPORTED_TARGETS})
