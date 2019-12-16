cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
add_compile_options(-std=c++11)
project(cuda_downsample) 
set(CMAKE_BUILD_TYPE "Release")
find_package(catkin REQUIRED)
find_package(PCL 1.5 REQUIRED)
find_package(CUDA 7.5 REQUIRED)


set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
catkin_package(
    DEPENDS PCL                                #Non-catkin CMake projects
    INCLUDE_DIRS include                        #The exported include paths
    LIBRARIES cuda_downsample                      #The exported libraries from the project
)
include_directories(include
${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_downsample/

)

include_directories(${PCL_INCLUDE_DIRS})
include_directories     (${CUDA_INCLUDE_DIRS}) 

add_definitions(${PCL_DEFINITIONS})

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50;-std=c++11; -Xptxas -v")


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;--ptxas-options=-v)


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
