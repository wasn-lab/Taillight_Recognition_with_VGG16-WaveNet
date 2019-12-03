cmake_minimum_required(VERSION 2.8.3)
project(pretrain_squseg)
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS libs)
find_package(PCL 1.8 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

catkin_package(
#  INCLUDE_DIRS src
#  LIBRARIES preprolib_squseg
#  CATKIN_DEPENDS pcl_ros roscpp sensor_msgs
  DEPENDS PCL 1.8
)

 
include_directories(
  ${catkin_INCLUDE_DIRS}
)

# file(GLOB_RECURSE SOURCES_SRCS *.cpp)

add_executable (${PROJECT_NAME} src/pretrain_squeezeseg)
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries (${PROJECT_NAME} ${catkin_LIBRARIES})

add_executable (${PROJECT_NAME}_v2 src/pretrain_squeezeseg_v2)
add_dependencies(${PROJECT_NAME}_v2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries (${PROJECT_NAME}_v2 ${catkin_LIBRARIES})

add_executable (output_XYZIL src/output_XYZIL)
add_dependencies(output_XYZIL ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries (output_XYZIL ${catkin_LIBRARIES})
