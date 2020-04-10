cmake_minimum_required(VERSION 3.1)
project(itri_pedcross)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})

include(CarModel)
if (NOT ENABLE_PEDCROSS)
  message(STATUS "${PROJECT_NAME} is disabled in car ${CAR_MODEL}")
  return ()
endif ()

include(CompilerFlags)

set(PED_MODEL_DIR ${Project_SOURCE_DIR}/sensing/itri_pedcross/models)

set(TENSORFLOW_ROOT_DIR ${Project_SOURCE_DIR}/dl_data/assets)
set(TENSORFLOW_INCLUDE_DIR ${TENSORFLOW_ROOT_DIR}/include)
set(TENSORFLOW_LIBRARIES tensorflow)
message("TF found: ${TENSORFLOW_LIBRARIES}")

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
    ${TENSORFLOW_INCLUDE_DIR}
    src/
)
link_directories(${TENSORFLOW_ROOT_DIR}/lib)

set(CPP_FILES
    src/pedestrian_event.cpp
    src/buffer.cpp
    src/buffer_node.cpp
    src/openpose.cpp
    src/openpose_flags.cpp
    src/openpose_ros_io.cpp
    #src/tf_utils.cpp
    #src/Predict_test.cpp
)


add_executable(itri_pedcross ${CPP_FILES})
target_link_libraries(
    itri_pedcross
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${openpose_LIBRARIES}
    ${GFLAGS_LIBRARY}
    ${GLOG_LIBRARY}
    ${TENSORFLOW_LIBRARIES})
add_dependencies(itri_pedcross ${catkin_EXPORTED_TARGETS})
configure_file(include/ped_def.h.in
               ${PROJECT_BINARY_DIR}/include/ped_def.h)