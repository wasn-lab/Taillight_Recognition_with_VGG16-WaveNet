cmake_minimum_required(VERSION 2.8.3)
project(localization)

find_package(PCL 1.8.1 REQUIRED)
find_package(CUDA)
if (NOT ${CUDA_FOUND})
  message("CUDA not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_generation
  genmsg
  roslib
  tf
  ndt_gpu
  cuda_downsample
  dynamic_reconfigure
)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(Eigen3)

add_message_files(
  FILES
  VehInfo.msg
  LocalizationToVeh.msg
  ErrorCode.msg
)


generate_messages(
  DEPENDENCIES
  std_msgs
)

generate_dynamic_reconfigure_options(
 cfg/localization.cfg
 )

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  roscpp
  std_msgs
  ndt_gpu
  cuda_downsample

)



include_directories(
 include
${catkin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
)


link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})



add_executable(${PROJECT_NAME}_node
    src/localization_main.cpp
)

target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
#target_link_libraries(dynamic_reconfigure_node ${catkin_LIBRARIES})

MESSAGE("Generate ${PROJECT_NAME}")

add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})


if(CUDA_FOUND)
  target_include_directories(${PROJECT_NAME}_node PRIVATE ${CUDA_INCLUDE_DIRS})

endif()
