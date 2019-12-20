if (CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  set(TEST_BAG "my_record_2019-11-27-16-36-54.bag")

  catkin_download_test_data(${TEST_BAG}
    "http://nas.itriadv.co:8888/Share/ADV/Rosbag/B1/20191127_in itri_verification/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 05e196fb30c6e04ffb029a48d788fb8e
  )
  # tests of drivenet
  add_rostest(drivenet_publish_detect_image.test
    DEPENDENCIES drivenet_node drivenet_node1 drivenet_node2 ${TEST_BAG})

#  Directly use the following tests in CI.
#  add_rostest(drivenet_60_hz.test
#    DEPENDENCIES drivenet_node ${TEST_BAG})
#  add_rostest(drivenet_120_1_hz.test
#    DEPENDENCIES drivenet_node1 ${TEST_BAG})
#  add_rostest(drivenet_120_2_hz.test
#    DEPENDENCIES drivenet_node2 ${TEST_BAG})
#  add_rostest(drivenet_hz.test
#    DEPENDENCIES drivenet_node drivenet_node1 drivenet_node2 ${TEST_BAG})
endif()
