cmake_minimum_required(VERSION 3.1)
project(itri_pedcross)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

set(PED_MODEL_DIR ${Project_SOURCE_DIR}/sensing/itri_pedcross/models)

catkin_download_test_data(
    ped_models_rf_pose
    "http://nas-cht.itriadv.co:8888/Share/ADV/S3_git_data/ped_models/rf_3frames_normalization.yml"
    DESTINATION ${PED_MODEL_DIR}
    MD5 ff1ce2293391cf50f224edc76bf4f022
)
catkin_download_test_data(
    ped_models_rf
    "http://nas-cht.itriadv.co:8888/Share/ADV/S3_git_data/ped_models/rf.yml"
    DESTINATION ${PED_MODEL_DIR}
    MD5 d20e9859ff9ba625e2747578dae99cf2
)
catkin_download_test_data(
    ped_models_body25
    "http://nas-cht.itriadv.co:8888/Share/ADV/S3_git_data/ped_models/body_25/pose_iter_584000.caffemodel"
    DESTINATION ${PED_MODEL_DIR}/pose/body_25
    MD5 78287b57cf85fa89c03f1393d368e5b7
)

catkin_download_test_data(
    ped_models_body25_txt
    "http://nas-cht.itriadv.co:8888/Share/ADV/S3_git_data/ped_models/body_25/pose_deploy.prototxt"
    DESTINATION ${PED_MODEL_DIR}/pose/body_25
    MD5 451938635cf007327c3cf235ed483844
)

find_package(catkin REQUIRED COMPONENTS
    message_generation
    roscpp
    geometry_msgs
    tf2_geometry_msgs
    sensor_msgs
    std_msgs
    msgs
    cv_bridge
    image_transport)

find_package(OpenCV REQUIRED)
find_package(openpose)

if (NOT openpose_LIBRARY)
  message(STATUS "Openpose not found. Follow ${CMAKE_CURRENT_LIST_DIR}/install_openpose.sh to install it. Skip building ${PROJECT_NAME}.")
  return()
endif ()

find_package(GFlags)
if (NOT GFLAGS_FOUND)
  message(FATAL_ERROR "libgflags not found. Please run\n ")
endif ()

find_package(Glog)

find_package(cv_bridge REQUIRED
             PATHS /opt/ros/kinetic)

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
    ${CUDA_INCLUDE_DIRS}
    ${openpose_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CAFFE_INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIR}
    ${PROJECT_BINARY_DIR}/include
    src/
)

set(CPP_FILES
    src/pedestrian_event.cpp
    src/buffer.cpp
    src/buffer_node.cpp
    src/openpose.cpp
    src/openpose_flags.cpp
    src/openpose_ros_io.cpp
)

add_executable(itri_pedcross ${CPP_FILES})
target_link_libraries(
    itri_pedcross
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${openpose_LIBRARIES}
    ${GFLAGS_LIBRARY}
    ${GLOG_LIBRARY})
add_dependencies(itri_pedcross ${catkin_EXPORTED_TARGETS})
configure_file(include/ped_def.h.in
               ${PROJECT_BINARY_DIR}/include/ped_def.h)
