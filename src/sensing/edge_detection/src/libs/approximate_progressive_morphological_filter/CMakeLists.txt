cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(approximate_progressive_morphological_filter)

set(CMAKE_MODULE_PATH ${Project_SOURCE_DIR}/cmake)
include(CompilerFlags)


find_package(catkin REQUIRED)
find_package(PCL)

set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
catkin_package(
    DEPENDS PCL                                #Non-catkin CMake projects
    INCLUDE_DIRS include                        #The exported include paths
    LIBRARIES approximate_progressive_morphological_filter  #The exported libraries from the project
)
include_directories(include
${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_downsample/

)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
find_package(Eigen3)

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50;-std=c++11; -Xptxas -v")


FILE(GLOB_RECURSE CPP_LIST *.cpp)
#FILE(GLOB_RECURSE CU_LIST *.cu)
FILE(GLOB_RECURSE H_LIST ${CMAKE_CURRENT_SOURCE_DIR}/include/approximate_progressive_morphological_filter/*.h
      ${CMAKE_CURRENT_SOURCE_DIR}/include/approximate_progressive_morphological_filter/*.hpp)


MESSAGE(STATUS "===================cpp list: ${CPP_LIST}")
#MESSAGE(STATUS "===================cu list: ${CU_LIST}")
MESSAGE(STATUS "===================h list: ${H_LIST}")


add_library(${PROJECT_NAME}
  ${CPP_LIST}
)
target_link_libraries(approximate_progressive_morphological_filter
  ${PCL_LIBRARIES}
        ${CUDA_LIBRARIES}
)

    install(DIRECTORY include/${PROJECT_NAME}/
            DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
            FILES_MATCHING PATTERN "*.h"
            )


    install(TARGETS approximate_progressive_morphological_filter
            ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
            RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
            )
