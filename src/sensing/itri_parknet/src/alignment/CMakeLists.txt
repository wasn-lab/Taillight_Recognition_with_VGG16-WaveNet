if (NOT ENABLE_PARKNET_ALIGNMENT_NODE)
  return()
endif ()

set(ALIGNMENT_LIB "parknet_alignment")

# Use jsoncpp to dump image_point -> spatial point mapping
find_package(jsoncpp REQUIRED)

find_library(JSONCPP_LIBRARY
  NAMES jsoncpp
  # PATHS /usr/lib/x86_64-linux-gnu
)
if (${JSONCPP_LIBRARY} STREQUAL "JSONCPP_LIBRARY-NOTFOUND")
  message(FATAL_ERROR "cannot find jsoncpp. Try apt-get install libjsoncpp-dev")
else ()
  message("Find libjson: ${JSONCPP_LIBRARY}")
endif ()

find_path(JSONCPP_INCLUDE_DIR
  NAMES json/json.h
  PATHS /usr/include/jsoncpp
)
if (${JSONCPP_INCLUDE_DIR} STREQUAL "JSONCPP_INCLUDE_DIR-NOTFOUND")
  message(FATAL_ERROR "cannot find jsoncpp headers. Try apt-get install libjsoncpp-dev")
else ()
  message("Find jsoncpp headers: ${JSONCPP_INCLUDE_DIR}")
endif ()

# Setup *.cpp to build libparknet
set(ALIGNMENT_LIB_SRCS
    Alignment.cpp
    alignment_params.cpp
    alignment_args_parser.cpp
    parknet_alignment_node.cpp
    alignment_json_writer.cpp
    alignment_json_reader.cpp
    alignment_utils.cpp)

set(ALIGNMENT_INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}
    ${catkin_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIR}
    ${PROJECT_BINARY_DIR}/include)

# set up dependencies for pslot_detector
set(ALIGNMENT_LIB_DEPS
    ${PCL_IO_LIBRARY}
    ${PCL_COMMON_LIBRARY}
    ${JSONCPP_LIBRARY}
    m pthread stdc++
    parknet
    ${catkin_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${GFLAGS_LIBRARIES})

include_directories(${ALIGNMENT_INCLUDE_DIRS})

add_library(${ALIGNMENT_LIB} STATIC ${ALIGNMENT_LIB_SRCS})
set_target_properties(${ALIGNMENT_LIB} PROPERTIES EXCLUDE_FROM_ALL TRUE)
target_link_libraries(${ALIGNMENT_LIB} ${ALIGNMENT_LIB_DEPS})

add_executable(parknet_alignment_node main.cpp)
target_link_libraries(parknet_alignment_node ${ALIGNMENT_LIB})
set_target_properties(parknet_alignment_node PROPERTIES EXCLUDE_FROM_ALL TRUE)

catkin_add_gtest(parknet_alignment_test
                 alignment_main_test.cpp
                 alignment_params_test.cpp
                 parknet_alignment_node_test.cpp
                 alignment_json_writer_test.cpp
                 alignment_json_reader_test.cpp
                 alignment_utils_test.cpp)
target_link_libraries(parknet_alignment_test ${ALIGNMENT_LIB})
