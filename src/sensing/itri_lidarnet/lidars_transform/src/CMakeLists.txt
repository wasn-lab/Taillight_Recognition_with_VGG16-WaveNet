set(PC_TRANSFORM_LIB "pc_transform")

set(PC_TRANSFORM_LIB_SRCS
  transform_points.cu
  pc_transform_args_parser.cpp
  pc_transform_node.cpp
  pc_transform_cpu.cpp
  pc_transform_gpu.cpp)

set(PC_TRANSFORM_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/include
    ${catkin_INCLUDE_DIRS}
    ${PCL_IO_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS})

# set up dependencies for pslot_detector
set(PC_TRANSFORM_LIB_DEPS
    m pthread stdc++ z
    ${GFLAGS_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${Boost_LIBRARIES}
    ${catkin_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${PCL_COMMON_LIBRARY})

include_directories(${PC_TRANSFORM_INCLUDE_DIRS})

cuda_add_library(${PC_TRANSFORM_LIB} STATIC ${PC_TRANSFORM_LIB_SRCS})
target_link_libraries(${PC_TRANSFORM_LIB} ${PC_TRANSFORM_LIB_DEPS})
add_dependencies(${PC_TRANSFORM_LIB} ${catkin_EXPORTED_TARGETS})

set(PC_TRANSFORM_NODE_EXE pc_transform_node)
set(PC_TRANSFORM_NODE_SRCS
  pc_transform_main.cpp)
add_executable(${PC_TRANSFORM_NODE_EXE} ${PC_TRANSFORM_NODE_SRCS})
target_link_libraries(${PC_TRANSFORM_NODE_EXE} ${PC_TRANSFORM_LIB} ${catkin_LIBRARIES})
add_dependencies(${PC_TRANSFORM_NODE_EXE} ${PC_TRANSFORM_LIB})


add_subdirectory(test)
