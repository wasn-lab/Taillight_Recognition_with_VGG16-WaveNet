cmake_minimum_required(VERSION 2.8.3)
project(drivenet_lib)

add_definitions(-O3 -g)

add_compile_options(-std=c++11) 

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
) 

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp rospy
)

find_package(OpenCV 3 REQUIRED PATHS /usr/local NO_DEFAULT_PATH)
if(OpenCV_FOUND)
  message(STATUS "OpenCV IncludeDir: ${OpenCV_INCLUDE_DIRS}")
  message(STATUS "OpenCV Libararies: ${OpenCV_LIBRARIES}")
  message(STATUS "OpenCV Libararies Dir: ${OpenCV_LIBRARY_DIRS}")
endif()

find_package(CUDA 10.0 REQUIRED)
if (CUDA_FOUND)
  message(STATUS "CUDA Include Dir: ${CUDA_INCLUDE_DIRS}")
  message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
  set(CUDA_LIBRARY_DIRS /usr/local/cuda/lib64) 
  message(STATUS "CUDA Libararies Dir: ${CUDA_LIBRARY_DIRS}")
  set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    -gencode arch=compute_30,code=sm_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=[sm_50,compute_50]
    -gencode arch=compute_52,code=[sm_52,compute_52]
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_62,code=sm_62
  )
endif()

find_package(TensorRT)
message(STATUS "TensorRT_LIBRARY: ${TensorRT_LIBRARY}")

find_package(PkgConfig REQUIRED)
pkg_search_module(GLIB REQUIRED glib-2.0)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LIBS ${OpenCV_LIBRARIES}
         nvinfer
         nvinfer_plugin
         ${CUDA_LIBRARIES}
         ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppisu.so
         ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppig.so
         ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppidei.so
         ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppial.so
         cuda
         glog
         )
add_definitions("-DNPP") 

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${GLIB_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${TrtYolo_LIBRARY_DIRS}
)
link_directories(
  ${CUDA_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
)
set(DRIVENETLIB_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include PARENT_SCOPE)
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}/include")
message(STATUS "DRIVENETLIB_INCLUDE: ${DRIVENETLIB_INCLUDE}")

file(GLOB_RECURSE LIB_SOURCES *.cu *.cpp)
message(STATUS "LIB_SOURCES: ${LIB_SOURCES}")
cuda_add_library(drivenet_lib SHARED ${LIB_SOURCES})
target_link_libraries(drivenet_lib ${LIBS} ${catkin_LIBRARIES} ${GLIB_LDFLAGS})
add_dependencies(drivenet_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
