include_directories(${IMAGE_COMPRESSOR_INCLUDE_DIRS})
# message(FATAL_ERROR "google_benchmark_SOURCE_DIR: ${google_benchmark_SOURCE_DIR}")
catkin_add_gtest(image_compressor_test
  image_compressor_test_main.cpp
  image_compressor_test.cpp)
catkin_download_test_data(
  608x384.png
  http://nas.itriadv.co:8888/git_data/B1/tests/608x384.png
  MD5 db5a175267aeb50ee698a66353335016)

target_link_libraries(image_compressor_test ${IMAGE_COMPRESSOR_LIB})
add_dependencies(image_compressor_test ${IMAGE_COMPRESSOR_LIB} ${catkin_EXPORTED_TARGETS})
