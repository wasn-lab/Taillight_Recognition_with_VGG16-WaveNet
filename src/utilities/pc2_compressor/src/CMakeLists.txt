set(PC2_COMPRESSOR_LIB "pc2_compressor")

set(PC2_COMPRESSOR_LIB_SRCS
    pc2_args_parser.cpp
    itri_pcd_reader.cpp
    itri_pcd_writer.cpp
    filter_node.cpp
    pc2_compressor.cpp)

set(PC2_COMPRESSOR_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/include
    ${catkin_INCLUDE_DIRS}
    ${PCL_IO_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIRS})

# set up dependencies for pslot_detector
set(PC2_COMPRESSOR_LIB_DEPS
    m pthread stdc++ z
    ${GFLAGS_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${Boost_LIBRARIES}
    ${catkin_LIBRARIES}
    ${Snappy_LIBRARIES}
    ${PCL_IO_LIBRARY}
    ${PCL_COMMON_LIBRARY})

include_directories(${PC2_COMPRESSOR_INCLUDE_DIRS})

add_library(${PC2_COMPRESSOR_LIB} STATIC ${PC2_COMPRESSOR_LIB_SRCS})
target_link_libraries(${PC2_COMPRESSOR_LIB} ${PC2_COMPRESSOR_LIB_DEPS})
add_dependencies(${PC2_COMPRESSOR_LIB} ${catkin_EXPORTED_TARGETS})

set(PC2_COMPRESSOR_NODE_EXE pc2_compressor_node)
set(PC2_COMPRESSOR_NODE_SRCS
  pc2_compressor_node.cpp
  cmpr_main.cpp)
add_executable(${PC2_COMPRESSOR_NODE_EXE} ${PC2_COMPRESSOR_NODE_SRCS})
target_link_libraries(${PC2_COMPRESSOR_NODE_EXE} ${PC2_COMPRESSOR_LIB} ${catkin_LIBRARIES})
add_dependencies(${PC2_COMPRESSOR_NODE_EXE} ${PC2_COMPRESSOR_LIB})

set(PC2_DECOMPRESSOR_NODE_EXE pc2_decompressor_node)
set(PC2_DECOMPRESSOR_NODE_SRCS
  pc2_decompressor_node.cpp
  decmpr_main.cpp)
add_executable(${PC2_DECOMPRESSOR_NODE_EXE} ${PC2_DECOMPRESSOR_NODE_SRCS})
target_link_libraries(${PC2_DECOMPRESSOR_NODE_EXE} ${PC2_COMPRESSOR_LIB} ${catkin_LIBRARIES})
add_dependencies(${PC2_DECOMPRESSOR_NODE_EXE} ${PC2_COMPRESSOR_LIB})

set(PC2_COMPRESSOR_TEST_DIR "${CMAKE_CURRENT_LIST_DIR}/test")
configure_file(pc2_compressor_def.h.in ${PROJECT_BINARY_DIR}/include/pc2_compressor_def.h)


add_subdirectory(test)
