if (CATKIN_ENABLE_TESTING)
  include_directories(${IMAGE_COMPRESSOR_INCLUDE_DIRS})
   catkin_download_test_data(
    608x384.png
    http://nas.itriadv.co:8888/git_data/B1/tests/608x384.png
    DESTINATION ${image_compressor_SOURCE_DIR}/src/test
    MD5 db5a175267aeb50ee698a66353335016)

  catkin_download_test_data(
    auto_record_2020-03-10-10-48-39_41_image_raw.bag
    http://nas.itriadv.co:8888/Share/ADV/Rosbag/B1/test_case/auto_record_2020-03-10-10-48-39_41_image_raw.bag
    DESTINATION ${image_compressor_SOURCE_DIR}/src/test
    MD5 90e4fdca30337c87ed1b55327a9f9e86)

  catkin_download_test_data(
    auto_record_2020-12-28-16-55-14_29_jpg.bag
    http://nas.itriadv.co:8888/Share/ADV/Rosbag/B1/test_case/auto_record_2020-12-28-16-55-14_29_jpg.bag
    DESTINATION ${image_compressor_SOURCE_DIR}/src/test
    MD5 b330372fe8f415c2961f8334a4ab31a0)

  catkin_add_gtest(image_compressor_test
    image_compressor_test_main.cpp
    image_compressor_test.cpp)

  target_link_libraries(image_compressor_test ${IMAGE_COMPRESSOR_LIB})
  add_dependencies(image_compressor_test ${IMAGE_COMPRESSOR_LIB} ${catkin_EXPORTED_TARGETS} 608x384.png)
endif()
