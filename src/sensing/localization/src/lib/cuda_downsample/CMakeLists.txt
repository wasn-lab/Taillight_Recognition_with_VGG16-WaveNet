cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(cuda_downsample)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)

find_package(catkin REQUIRED)
find_package(PCL 1.5 REQUIRED)

find_package(CUDA)
if (NOT ${CUDA_FOUND})
  message("CUDA not found. Skip building ${PROJECT_NAME}")
  return ()
endif ()

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
catkin_package(
    DEPENDS PCL                                #Non-catkin CMake projects
    INCLUDE_DIRS include                        #The exported include paths
    LIBRARIES cuda_downsample                      #The exported libraries from the project
)
include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_downsample)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

add_definitions(${PCL_DEFINITIONS})

find_package(Eigen3)


FILE(GLOB_RECURSE CPP_LIST *.cpp)
FILE(GLOB_RECURSE CU_LIST *.cu)
FILE(GLOB_RECURSE H_LIST ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_downsample/*.h
      ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_downsample/*.cuh)


MESSAGE(STATUS "cpp list: ${CPP_LIST}")
MESSAGE(STATUS "cu list: ${CU_LIST}")
MESSAGE(STATUS "h list: ${H_LIST}")


cuda_add_library(${PROJECT_NAME}
  ${CPP_LIST}
  ${CU_LIST}
)
target_link_libraries(cuda_downsample
  ${PCL_LIBRARIES}
        ${CUDA_LIBRARIES}
)

    install(DIRECTORY include/${PROJECT_NAME}/
            DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
            FILES_MATCHING PATTERN "*.h"
            )


    install(TARGETS cuda_downsample
            ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
            )
