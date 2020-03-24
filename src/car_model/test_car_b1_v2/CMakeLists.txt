if (CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  set(TEST_BAG "auto_record_2020-03-10-10-48-39_41.bag")

  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/20200310_in_shalun/pp_scenario1/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 d15fa88484359d004aec43ec10ec43d0
  )
  # tests of drivenet
  add_rostest(drivenet_publish_detect_image.test
    DEPENDENCIES drivenet_node_a drivenet_node_b drivenet_node_c ${TEST_BAG})
endif()
