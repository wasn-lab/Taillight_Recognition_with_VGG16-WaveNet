cmake_minimum_required(VERSION 2.8.3)
project(drivenet)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)                              
include(CompilerFlags)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  msgs
  image_transport
  cv_bridge
  car_model
  camera_utils
  drivenet_lib
)
set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)

find_package(OpenCV 3 REQUIRED PATHS /usr/local NO_DEFAULT_PATH)
find_package(CUDA REQUIRED)
find_package(TensorRT)
message(STATUS "TensorRT_LIBRARY: ${TensorRT_LIBRARY}")

catkin_package(
  INCLUDE_DIRS include
)

set(LIBS drivenet_lib
         camera_utils
         ${OpenCV_LIBRARIES}
         nvinfer
         nvinfer_plugin
         ${CUDA_LIBRARIES}
         ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppisu.so
         ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppig.so
         gflags 
         stdc++fs)

add_definitions("-DNPP") 

include_directories(
  include 
  ${INCLUDE_DIRS}
  ${car_model_BINARY_DIR}/include
  ${camera_utils_SOURCE_DIR}/include
  ${DRIVENETLIB_INCLUDE}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)
message(STATUS "DRIVENETLIB_INCLUDE: ${DRIVENETLIB_INCLUDE}")

link_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDA_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
)

file(GLOB_RECURSE EXCEPTION_SRCS src/exception_handling/*.cpp)

add_executable(${PROJECT_NAME}_node src/drivenet_60_b1.cpp  src/trt_yolo_interface.cpp  src/distance_estimation_b1.cpp src/object_label_util.cpp src/boundary_util.cpp src/image_preprocessing.cpp  ${EXCEPTION_SRCS})
add_executable(${PROJECT_NAME}_node1 src/drivenet_120_1_b1.cpp  src/trt_yolo_interface.cpp src/distance_estimation_b1.cpp src/object_label_util.cpp src/boundary_util.cpp  src/image_preprocessing.cpp   ${EXCEPTION_SRCS})
add_executable(${PROJECT_NAME}_node2 src/drivenet_120_2_b1.cpp  src/trt_yolo_interface.cpp src/distance_estimation_b1.cpp src/object_label_util.cpp src/boundary_util.cpp  src/image_preprocessing.cpp  ${EXCEPTION_SRCS})

add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(${PROJECT_NAME}_node1 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(${PROJECT_NAME}_node2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
   ${LIBS}
)

target_link_libraries(${PROJECT_NAME}_node1
   ${catkin_LIBRARIES}
   ${LIBS}
)

target_link_libraries(${PROJECT_NAME}_node2
   ${catkin_LIBRARIES}
   ${LIBS}
)
