cmake_minimum_required(VERSION 2.8.3)
project(convex_fusion)
 
set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  msgs
  camera_utils
)  

find_package(libs)                
if (NOT ${libs_FOUND})
  message("libs not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()

catkin_package(
  CATKIN_DEPENDS roscpp rospy msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
 include
  ${catkin_INCLUDE_DIRS}
  ${car_model_BINARY_DIR}/include
  ${camera_utils_SOURCE_DIR}/include
  ${libs_SOURCE_DIR}
  ${libs_SOURCE_DIR}/Preprocess/VoxelGrid
  ${libs_SOURCE_DIR}/Preprocess/NoiseFilter
  ${libs_SOURCE_DIR}/BoundingBox/ApproxMVBB
)

######################
##       PCL        ##
######################
find_package            (PCL 1.8 REQUIRED) 
include_directories     (${PCL_INCLUDE_DIRS})
link_directories        (${PCL_LIBRARY_DIRS})
add_definitions         (${PCL_DEFINITIONS})
include_directories     (${CUDA_INCLUDE_DIRS}) 
message("CUDA found: ${CUDA_FOUND} CUDA version: ${CUDA_VERSION_STRING}")
   
######################
##   Source file    ##
######################
file(GLOB_RECURSE SOURCES_SRCS *.cpp *.cu)
list(REMOVE_ITEM SOURCES_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/convex_fusion_hino.cpp")
list(REMOVE_ITEM SOURCES_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/convex_fusion_b1.cpp")
 
######################
##       CUDA       ##
######################
find_package	        (CUDA REQUIRED) 
if (NOT ${CUDA_FOUND})
  message("CUDA not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()
set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
 
######################
##     Vehicle      ##
######################    
cuda_add_executable   (${PROJECT_NAME}_hino src/convex_fusion_hino.cpp ${SOURCES_SRCS})
add_dependencies      (${PROJECT_NAME}_hino msgs_generate_messages_cpp ${catkin_EXPORTED_TARGETS}) 
target_link_libraries (${PROJECT_NAME}_hino ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${CUDA_LIBRARIES} cuda libs)
qt5_use_modules       (${PROJECT_NAME}_hino Widgets)

cuda_add_executable   (${PROJECT_NAME}_b1 src/convex_fusion_b1.cpp ${SOURCES_SRCS})
add_dependencies      (${PROJECT_NAME}_b1 msgs_generate_messages_cpp ${catkin_EXPORTED_TARGETS}) 
target_link_libraries (${PROJECT_NAME}_b1 ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${CUDA_LIBRARIES} cuda camera_utils libs)
qt5_use_modules       (${PROJECT_NAME}_b1 Widgets)