if (CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)

  set(TEST_BAG "auto_record_2020-03-10-10-48-39_41.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 d15fa88484359d004aec43ec10ec43d0
  )

  set(TEST_BAG "auto_record_2020-04-14-16-41-15_89.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 14e46bab793d858db78b52213b61e5a0
  )

  set(TEST_BAG "lidar_raw_2020-03-10-10-48-39_41.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 e2ff37c83b2e95aebeff9b192cf88c64
  )

  set(TEST_BAG "edge_detection_2020-04-13-17-45-48_0.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 fccf1132a9058350449afe2d32502f64
  )

  set(TEST_BAG "localization_raw_2020-04-13-17-45-48_0.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 16ef3be30e7e2d06b1119021abdff31a
  )

  set(TEST_BAG "lidarxyz2lla_2020-04-13-17-45-48_0.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 1439b6181d4282492b9046a434eeba21
  )

  set(TEST_BAG "rad_grab_2020-04-13-17-45-48_0.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 f215af4ad6b04fa9ae152f8b46b0f2d8
  )

  set(TEST_BAG "ukf_mm_2020-04-13-17-45-48_0.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 5e614042fed2962b56161c3b12c138db
  )

  set(TEST_BAG "target_planner_2020-04-13-17-45-48_0.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 fe9ddeb4a65c5c518145a2a5edd4dd08
  )

  set(TEST_BAG "auto_record_2020-06-19-16-26-18_1_filtered.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 93e265acc97c7f951930f01d7cb938a5
  )

  set(TEST_BAG "auto_record_2020-08-04-10-15-25_4_filtered.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 8c8364b3398751ce2acb0717451029e0
  )

endif()
