
find_package(Benchmark)
include_directories(${PC2_COMPRESSOR_INCLUDE_DIRS} ${benchmark_INCLUDE_DIRS})

#catkin_download_test_data(velodyne32.pcd
#  "http://118.163.54.109:8888/git_data/B1/tests/velodyne32.pcd"
#  DESTINATION ${pc2_compressor_SOURCE_DIR}/src/test
#  MD5 3826f9e2bb13e17c12a63e708d88ed1d)

catkin_download_test_data(ouster64.pcd
  "http://118.163.54.109:8888/git_data/B1/tests/ouster64.pcd"
  DESTINATION ${pc2_compressor_SOURCE_DIR}/src/test
  MD5 14c96d5bdf4d3d80ccfc9d309d90be8a)

catkin_add_gtest(pc2_compressor_test
  pc2_compressor_test_main.cpp
  pc2_compressor_test_utils.cpp
  pc2_compressor_test.cpp)

target_link_libraries(pc2_compressor_test ${PC2_COMPRESSOR_LIB})
add_dependencies(pc2_compressor_test ${PC2_COMPRESSOR_LIB} ${catkin_EXPORTED_TARGETS} ouster64.pcd)

if (benchmark_FOUND)
  add_executable(pc2_compressor_perf pc2_compressor_perf.cpp pc2_compressor_test_utils.cpp)
  target_link_libraries(pc2_compressor_perf ${PC2_COMPRESSOR_LIB} ${benchmark_LIBRARIES})
  add_dependencies(pc2_compressor_perf ${PC2_COMPRESSOR_LIB} ${catkin_EXPORTED_TARGETS})
endif ()
