cmake_minimum_required(VERSION 3.1)
project(itri_gmphd)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

set(CATKIN_REQUIRED_COMPONENTS
    roscpp
    sensor_msgs
    std_msgs
    msgs
    car_model
    cv_bridge
    image_transport
    )
find_package(catkin REQUIRED COMPONENTS ${CATKIN_REQUIRED_COMPONENTS})

set(GMPHD_DIR ${Project_SOURCE_DIR}/sensing/itri_gmphd/)

find_package(OpenCV
  PATHS /usr/local
  NO_DEFAULT_PATH
)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost REQUIRED COMPONENTS thread system wave timer signals serialization regex random program_options mpi log log_setup locale iostreams graph graph_parallel filesystem exception date_time coroutine context chrono atomic)

set(TPP_CATKIN_DEPS
    roscpp
    sensor_msgs
    std_msgs
    msgs
)

catkin_package(CATKIN_DEPENDS ${TPP_CATKIN_DEPS})

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CAFFE_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRs}
    ${PROJECT_BINARY_DIR}/include
    src/
)

set(CPP_FILES
    src/ros_gmphd.cpp
    src/demo_GMPHD-OGM.cpp
    src/GMPHD_OGM.cpp
    src/HungarianAlgorithm.cpp
    src/stdafx.cpp
)

add_executable(itri_gmphd ${CPP_FILES})
add_executable(bag_to_MOT src/bag_to_MOT.cpp)

if(EXISTS /usr/local/lib/libopencv_core.so)
target_link_libraries(
    itri_gmphd
    /usr/local/lib/libopencv_core.so
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES})
else()
target_link_libraries(
    itri_gmphd
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES})
endif()
if(EXISTS /usr/local/lib/libopencv_core.so)
target_link_libraries(
    bag_to_MOT
    /usr/local/lib/libopencv_core.so
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES})
else()
target_link_libraries(
    bag_to_MOT
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES})
endif()

add_dependencies(itri_gmphd ${catkin_EXPORTED_TARGETS})
add_dependencies(bag_to_MOT ${catkin_EXPORTED_TARGETS})

configure_file(include/gmphd_def.h.in
               ${PROJECT_BINARY_DIR}/include/gmphd_def.h)
