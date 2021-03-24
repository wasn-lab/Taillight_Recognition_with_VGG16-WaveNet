
find_package(Benchmark)
include_directories(${PC2_COMPRESSOR_INCLUDE_DIRS} ${benchmark_INCLUDE_DIRS})

catkin_download_test_data(ouster64.pcd
  "http://nas.itriadv.co:8888/git_data/B1/tests/ouster64.pcd"
  DESTINATION ${pc2_compressor_SOURCE_DIR}/src/test
  MD5 14c96d5bdf4d3d80ccfc9d309d90be8a)

# Already defined in car_model package.
#set(TEST_BAG "lidar_raw_2020-12-28-16-53-14_21.bag")
#catkin_download_test_data(${TEST_BAG}
#  "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
#  DESTINATION ${Project_SOURCE_DIR}/bags
#  MD5 a94d0f1f4ccf9a490dd78f7b7d72b1bb
#)

catkin_download_test_data(lidar_compressed_2020-03-10-10-48-39_41.bag
  "http://nas.itriadv.co:8888/Share/ADV/Rosbag/B1/test_case/lidar_compressed_2020-03-10-10-48-39_41.bag"
  DESTINATION ${pc2_compressor_SOURCE_DIR}/src/test
  MD5 e62b87e69f9a96ceaf60f81999463f1d)

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
