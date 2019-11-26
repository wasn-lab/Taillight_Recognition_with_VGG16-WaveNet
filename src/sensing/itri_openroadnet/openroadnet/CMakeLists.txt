cmake_minimum_required(VERSION 2.8.3)
project(openroadnet)
find_package(OpenCV 3 REQUIRED PATHS /usr/local NO_DEFAULT_PATH)
## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  msgs
  libs_opn
  image_transport
  message_generation
  cv_bridge
)

#generate_messages(DEPENDENCIES std_msgs)



######################
##       CUDA       ##
######################
find_package	        (CUDA REQUIRED) 
set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
include_directories     (${CUDA_INCLUDE_DIRS}) 
set(CUDA_LIBS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppisu.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppig.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppc.so
                #${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppial.so
                #${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppicc.so
                #${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppicom.so
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppidei.so
                ${CUDA_cublas_LIBRARY}
                ${CUDA_curand_LIBRARY})
message("CUDA found: ${CUDA_FOUND} CUDA version: ${CUDA_VERSION_STRING}")


catkin_package(
   INCLUDE_DIRS src
#  LIBRARIES openroadnet
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
   CATKIN_DEPENDS msgs libs_opn
)

###########
## Build ##
###########
set (CMAKE_BUILD_TYPE Release) 
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -D_FORCE_INLINES")
set(OpenCV_INCLUDE_DIRS "/usr/local/include")
set(OpenCV_LIBRARY_DIRS "/usr/local/lib")
set(OpenCV_LIBRARIES opencv_core 
                     opencv_imgproc 
                     opencv_imgcodecs 
                     opencv_highgui 
                     opencv_dnn 
                     opencv_videoio)

set(Tensorflow_INCLUDE "../libs_opn/TensorFlow/Installed/include/")

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
${CUDA_TOOLKIT_INCLUDE}
${Tensorflow_INCLUDE}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/openroadnet.cpp
# )

######################
##      weights     ##
######################

set(PLUGIN_SOURCES
src/openroadnet.cpp 
src/DistanceEstimation.cpp 
src/main.cpp
)




## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
#file(GLOB_RECURSE SOURCES_SRCS *.cpp *.cu)            
#cuda_add_library(${PROJECT_NAME} STATIC ${PLUGIN_SOURCES})


add_executable(openroadnet src/openroadnet.cpp src/DistanceEstimation.cpp src/main.cpp)
add_dependencies(openroadnet ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
#target_include_directories(openroadnet PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(openroadnet libs_opn ${catkin_LIBRARIES} ${CUDA_LIBRARIES} cuda ${CUDA_LIBS})

add_library(openroadnet_lib src/openroadnet.cpp src/DistanceEstimation.cpp)
add_dependencies(openroadnet_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(openroadnet_lib libs_opn ${catkin_LIBRARIES})


