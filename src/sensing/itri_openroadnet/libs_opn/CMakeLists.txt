cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)
project(libs_opn)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})

include(CarModel)
if (NOT ENABLE_OPENROADNET)
  message(STATUS "${PROJECT_NAME} is disabled in car ${CAR_MODEL}")
  return ()
endif ()

include(CompilerFlags)

set(TENSORFLOW_ROOT_DIR ${Project_SOURCE_DIR}/dl_data/assets)
set(TENSORFLOW_INCLUDE_DIR ${TENSORFLOW_ROOT_DIR}/include/tensorflow/c)
set(TENSORFLOW_LIBRARIES ${TENSORFLOW_ROOT_DIR}/lib/libtensorflow.so)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  car_model
)

MACRO(HEADER_DIRECTORIES return_list)
    FILE(GLOB_RECURSE new_list *.h *.hpp)
    SET(dir_list "")
    FOREACH(file_path ${new_list})
        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
        SET(dir_list ${dir_list} ${dir_path})
    ENDFOREACH()
    LIST(REMOVE_DUPLICATES dir_list)
    SET(${return_list} ${dir_list})
ENDMACRO()

HEADER_DIRECTORIES(SOURCES_INCS)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${TENSORFLOW_INCLUDE_DIR}
)

######################
##       CUDA       ##
######################
find_package(CUDA)
if (NOT ${CUDA_FOUND})
  message("CUDA not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
include_directories     (${CUDA_INCLUDE_DIRS})
set(CUDA_LIBS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppisu.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppig.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppc.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppial.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppicc.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppicom.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppidei.so
                ${CUDA_cublas_LIBRARY}
                ${CUDA_curand_LIBRARY})
message("CUDA found: ${CUDA_FOUND} CUDA version: ${CUDA_VERSION_STRING}")

  #find_cuda_helper_libs(nppc)
  #find_cuda_helper_libs(nppial)
  #find_cuda_helper_libs(nppicc)
  #find_cuda_helper_libs(nppicom)
  #find_cuda_helper_libs(nppidei)
  #find_cuda_helper_libs(nppif)
  #find_cuda_helper_libs(nppig)
  #find_cuda_helper_libs(nppim)
  #find_cuda_helper_libs(nppist)
  #find_cuda_helper_libs(nppisu)
  #find_cuda_helper_libs(nppitc)
  #find_cuda_helper_libs(npps)



######################
##      OpenCV      ##
######################
find_package(OpenCV REQUIRED PATHS /usr/local NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

######################
##        ##
######################
catkin_package(
  INCLUDE_DIRS ${SOURCES_INCS} ${TENSORFLOW_INCLUDE_DIR}
  CATKIN_DEPENDS roscpp rospy
  LIBRARIES ${PROJECT_NAME}
)


######################
##      libs        ##
######################
file(GLOB_RECURSE SOURCES_SRCS *.cpp *.cu)
cuda_add_library(${PROJECT_NAME} ${SOURCES_SRCS})
target_link_libraries (${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} cuda ${TENSORFLOW_LIBRARIES} ${CUDA_LIBS})

