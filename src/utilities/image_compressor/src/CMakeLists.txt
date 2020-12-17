set(IMAGE_COMPRESSOR_LIB "image_compressor")

set(IMAGE_COMPRESSOR_LIB_SRCS
    image_compressor_args_parser.cpp
    image_compressor_priv.cpp
    image_compressor.cpp)

set(IMAGE_COMPRESSOR_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/include
    ${catkin_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS})

# set up dependencies for pslot_detector
set(IMAGE_COMPRESSOR_LIB_DEPS
    m pthread stdc++
    ${GFLAGS_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES})

include_directories(${IMAGE_COMPRESSOR_INCLUDE_DIRS})

add_library(${IMAGE_COMPRESSOR_LIB} STATIC ${IMAGE_COMPRESSOR_LIB_SRCS})
target_link_libraries(${IMAGE_COMPRESSOR_LIB} ${IMAGE_COMPRESSOR_LIB_DEPS})

set(IMAGE_COMPRESSOR_NODE_EXE image_compressor_node)
set(IMAGE_COMPRESSOR_NODE_SRCS
  image_compressor_node.cpp
  cmpr_main.cpp)
add_executable(${IMAGE_COMPRESSOR_NODE_EXE} ${IMAGE_COMPRESSOR_NODE_SRCS})
target_link_libraries(${IMAGE_COMPRESSOR_NODE_EXE} ${IMAGE_COMPRESSOR_LIB} ${catkin_LIBRARIES})

set(IMAGE_DECOMPRESSOR_NODE_EXE image_decompressor_node)
set(IMAGE_DECOMPRESSOR_NODE_SRCS
  image_decompressor_node.cpp
  decmpr_main.cpp)
add_executable(${IMAGE_DECOMPRESSOR_NODE_EXE} ${IMAGE_DECOMPRESSOR_NODE_SRCS})
target_link_libraries(${IMAGE_DECOMPRESSOR_NODE_EXE} ${IMAGE_COMPRESSOR_LIB} ${catkin_LIBRARIES})


add_subdirectory(test)
