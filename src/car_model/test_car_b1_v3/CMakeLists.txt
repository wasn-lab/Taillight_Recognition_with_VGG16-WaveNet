if (CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)

  set(TEST_BAG "camera_raw_2021-02-25-15-53-00_77.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 563baab43a945a286039be37711a26ec
  )

  set(TEST_BAG "alignment_auto_record_2021-04-22-23-13-32_27.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 a94bf732bd48ac0aea72ef3bbf1a36bb
  )

  set(TEST_BAG "pedcross_2021-05-06-13-36-41_0_filtered.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 24776ea8220a63d60ae1bd32eb73ddb7
  )

  set(TEST_BAG "auto_record_2020-04-14-16-41-15_89.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 14e46bab793d858db78b52213b61e5a0
  )

  set(TEST_BAG "geofence_pp_2020-11-16-16-35-39.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 62bf328edc7eb796366161e36c3f66eb
  )

  set(TEST_BAG "lidar_raw_compressed_2021-02-03.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 fe090879b354151dfa6c4674422d0964
  )

  set(TEST_BAG "lidar_detection_car_ped_cyc_2020-12-28-16-53-14_21.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 92839d7c373836381a0aa4ed82b0e3e3
  )

  set(TEST_BAG "lidar_raw_2020-12-28-16-53-14_21.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 a94d0f1f4ccf9a490dd78f7b7d72b1bb
  )


  set(TEST_BAG "localization_raw_2020-09-24-17-02-06.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 0aaf93949a9b8f20d93dd27d6672137b
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

  set(TEST_BAG "tracking_2d_2020-11-16-15-02-12.bag")
  catkin_download_test_data(${TEST_BAG}
    "http://118.163.54.109:8888/Share/ADV/Rosbag/B1/test_case/${TEST_BAG}"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 a7a3e9bc06292cccabe478ddb22a60a5
  )

endif()
