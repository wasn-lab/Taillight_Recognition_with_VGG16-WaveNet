cmake_minimum_required(VERSION 3.1)
project(itri_pedcross)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 11)  # -std=c++11
set(CMAKE_C_STANDARD 11)  # -std=gnu11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-parameter")

set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")

include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64") 

include_directories("/opt/ros/kinetic/include")
link_directories("/opt/ros/kinetic/lib") 

set(PED_MODEL_DIR ${Project_SOURCE_DIR}/sensing/itri_pedcross/models)

catkin_download_test_data(
    ped_models_rf
    "http://nas.itriadv.co:8888/Share/ADV/S3_git_data/ped_models/rf.yml"
    DESTINATION ${PED_MODEL_DIR}
    MD5 d20e9859ff9ba625e2747578dae99cf2
)
catkin_download_test_data(
    ped_models_mpi
    "http://nas.itriadv.co:8888/Share/ADV/S3_git_data/ped_models/mpi/pose_iter_160000.caffemodel"
    DESTINATION ${PED_MODEL_DIR}/mpi
    MD5 2ca0990c7562bd7ae03f3f54afa96e00
)

catkin_download_test_data(
    ped_models_mpi_txt
    "http://nas.itriadv.co:8888/Share/ADV/S3_git_data/ped_models/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    DESTINATION ${PED_MODEL_DIR}/mpi
    MD5 67ad3541c09b6f015151a230f52aa0b1
)

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

find_package(OpenCV 3.4.3 REQUIRED
             NO_MODULE
             PATHS /usr/local /opt/ros/kinetic
             NO_DEFAULT_PATH)

find_package(cv_bridge REQUIRED
             PATHS /opt/ros/kinetic)

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
        ${PROJECT_BINARY_DIR}/include
	src/
)

set(CPP_FILES
    src/pedestrian_event.cpp
    src/Get_keypoints_and_predict.cpp
)

add_executable(itri_pedcross ${CPP_FILES})
target_link_libraries(
        itri_pedcross
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        #caffe
	glog
        cv_bridge
)
add_dependencies(itri_pedcross ${catkin_EXPORTED_TARGETS})
configure_file(include/ped_def.h.in
               ${PROJECT_BINARY_DIR}/include/ped_def.h)

