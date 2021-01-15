
find_package(Benchmark)
include_directories(${PC2_COMPRESSOR_INCLUDE_DIRS} ${benchmark_INCLUDE_DIRS})
# message(FATAL_ERROR "google_benchmark_SOURCE_DIR: ${google_benchmark_SOURCE_DIR}")
catkin_add_gtest(pc2_compressor_test
  pc2_compressor_test_main.cpp
  pc2_compressor_test_utils.cpp
  pc2_compressor_test.cpp)

target_link_libraries(pc2_compressor_test ${PC2_COMPRESSOR_LIB})
add_dependencies(pc2_compressor_test ${PC2_COMPRESSOR_LIB} ${catkin_EXPORTED_TARGETS})

if (benchmark_FOUND)
  add_executable(pc2_compressor_perf pc2_compressor_perf.cpp pc2_compressor_test_utils.cpp)
  target_link_libraries(pc2_compressor_perf ${PC2_COMPRESSOR_LIB} ${benchmark_LIBRARIES})
  add_dependencies(pc2_compressor_perf ${PC2_COMPRESSOR_LIB} ${catkin_EXPORTED_TARGETS})
endif ()
