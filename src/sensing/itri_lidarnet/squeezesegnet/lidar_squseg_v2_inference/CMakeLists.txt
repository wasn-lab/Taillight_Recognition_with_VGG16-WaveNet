cmake_minimum_required(VERSION 2.8.3)
project(lidar_squseg_v2_inference)

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  msgs
  libs
  nodelet
  car_model
)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake ${car_model_DIR})
include(CompilerFlags)
include(CarModel)

######################
##       CUDA       ##
######################
find_package	        (CUDA)
if (NOT ${CUDA_FOUND})
  message("CUDA not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()
set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
include_directories     (${CUDA_INCLUDE_DIRS})
message("CUDA found: ${CUDA_FOUND} CUDA version: ${CUDA_VERSION_STRING}")

generate_messages(DEPENDENCIES std_msgs)

catkin_package(
#  INCLUDE_DIRS include
   LIBRARIES ${PROJECT_NAME}_nodelet
#  CATKIN_DEPENDS pcl_ros roscpp sensor_msgs
#  DEPENDS system_lib
   CATKIN_DEPENDS msgs
)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

######################
##       PCL        ##
######################
find_package            (PCL 1.8 REQUIRED)
include_directories     (${PCL_INCLUDE_DIRS})
link_directories        (${PCL_LIBRARY_DIRS})
add_definitions         (${PCL_DEFINITIONS})

include_directories(
  ${catkin_INCLUDE_DIRS}
)

file(GLOB_RECURSE SOURCES_SRCS *.cpp *.cu)
list(REMOVE_ITEM SOURCES_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/squeezeseg_inference.cpp")

add_executable		  (${PROJECT_NAME} src/squeezeseg_inference.cpp ${SOURCES_SRCS})
add_dependencies	  (${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries (${PROJECT_NAME} ${catkin_LIBRARIES} ${PCL_LIBRARIES})


######################
##      Nodelet     ##
######################
cuda_add_library	  (${PROJECT_NAME}_nodelet ${SOURCES_SRCS})
add_dependencies	  (${PROJECT_NAME}_nodelet ${${PROJECT_NAME}_nodelet_EXPORTED_TARGETS} msgs_generate_messages_cpp ${catkin_EXPORTED_TARGETS})
target_link_libraries (${PROJECT_NAME}_nodelet ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${CUDA_LIBRARIES} cuda)
