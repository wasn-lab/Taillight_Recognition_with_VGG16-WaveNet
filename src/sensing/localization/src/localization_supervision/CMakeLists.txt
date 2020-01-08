cmake_minimum_required(VERSION 2.8.3)
project(localization_supervision)
add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")


find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  roslib
  tf
  msgs
)

## Find Point Cloud Library 1.8
find_package(PCL 1.8.1 REQUIRED)



catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  roscpp
  std_msgs
  
  )

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(${PROJECT_NAME}_node src/localization_supervision.cpp)

target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})
add_dependencies(${PROJECT_NAME}_node ${PROJECT_NAME}_node_generate_messages_cpp)
MESSAGE("Generate ${PROJECT_NAME}")


