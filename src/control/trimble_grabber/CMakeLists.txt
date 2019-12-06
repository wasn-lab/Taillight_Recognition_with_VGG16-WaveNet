cmake_minimum_required(VERSION 2.8.3)
project(trimble_gps_imu_pub)
add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  roslib
  tf
)

#generate_messages(
#  DEPENDENCIES
#  std_msgs
#)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs
)

include_directories(
 include
  ${catkin_INCLUDE_DIRS}
)


add_executable(${PROJECT_NAME}_node src/GsofParser_imu_c.cpp)

target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})

MESSAGE("Generate ${PROJECT_NAME}")

add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})

add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS} )
#add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
