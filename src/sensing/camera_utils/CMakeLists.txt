cmake_minimum_required(VERSION 3.1)
project(camera_utils)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(catkin REQUIRED COMPONENTS
  car_model
  roscpp
  geometry_msgs)
catkin_package(CATKIN_DEPENDS roscpp geometry_msgs car_model)

find_package(CUDA)
if (NOT ${CUDA_FOUND})
  message("CUDA not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(GLM REQUIRED)

find_library(JSONCPP_LIBRARY
  NAMES jsoncpp
  # PATHS /usr/lib/x86_64-linux-gnu
)
if (NOT JSONCPP_LIBRARY)
  message(FATAL_ERROR "cannot find jsoncpp. Try apt-get install libjsoncpp-dev")
else ()
  message("Find libjson: ${JSONCPP_LIBRARY}")
endif ()

find_path(JSONCPP_INCLUDE_DIR
  NAMES json/json.h
  PATHS /usr/include/jsoncpp
)

if (NOT JSONCPP_INCLUDE_DIR)
  message(FATAL_ERROR "cannot find jsoncpp headers. Try apt-get install libjsoncpp-dev")
else ()
  message("Find jsoncpp headers: ${JSONCPP_INCLUDE_DIR}")
endif ()

set(CAMERA_UTILS_SOURCES
  src/camera_utils.cpp
  src/camera_params_hino.cpp
  src/camera_params_b1.cpp
  src/camera_params_b1_v2.cpp
  src/camera_params_b1_v3.cpp 
  src/camera_distance_mapper.cpp
  src/cv_color.cpp
  src/imu_distance.cpp
  src/imu_utils.cpp
  src/npp_remap.cpp
  src/npp_remapper.cpp
  src/npp_remapper_impl.cpp
  src/npp_resize.cpp
  src/npp_resizer.cpp
  src/npp_resizer_impl.cpp
  src/npp_rotate.cpp
  src/npp_utils.cpp
  src/projection/parameters.cpp
  src/projection/projector2.cpp
  src/projection/projector3.cpp)

set(CAMERA_UTILS_LIB_DEPS
  ${OpenCV_LIBS}
  ${GLOG_LIBRARIES}
	${JSONCPP_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppisu.so
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppidei.so
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppig.so)

set(CAMERA_UTILS_INCLUDE_DIRS
  src
  ${catkin_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_INCLUDE}
  ${GLOG_INCLUDE_DIRS}
	${JSONCPP_INCLUDE_DIR}
  ${car_model_BINARY_DIR}/include
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/projection)

include_directories(${CAMERA_UTILS_INCLUDE_DIRS})

add_library(${PROJECT_NAME} STATIC ${CAMERA_UTILS_SOURCES})
target_link_libraries(${PROJECT_NAME} ${CAMERA_UTILS_LIB_DEPS})
add_subdirectory(src/test)

set(CAMERA_UTILS_DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/data)
configure_file(include/camera_utils_defs.h.in ${PROJECT_BINARY_DIR}/include/camera_utils_defs.h)
