if (CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)

  set(TEST_BAG "auto_record_2020-03-10-10-48-39_41.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 d15fa88484359d004aec43ec10ec43d0
  )

  set(TEST_BAG "lidar_raw_2020-03-10-10-48-39_41.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 e2ff37c83b2e95aebeff9b192cf88c64
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
endif()
