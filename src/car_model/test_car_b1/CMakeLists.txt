if (CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)

  catkin_download_test_data(my_record_2019-11-27-16-36-54.bag
    "http://nas.itriadv.co:8888/Share/ADV/Rosbag/B1/20191127_in itri_verification/my_record_2019-11-27-16-36-54.bag"
    DESTINATION ${Project_SOURCE_DIR}/bags
    MD5 05e196fb30c6e04ffb029a48d788fb8e
  )
  add_rostest(b1_drivenet_60.test
    DEPENDENCIES drivenet_node drivenet_node1 drivenet_node2 drivenet_lib my_record_2019-11-27-16-36-54.bag)
endif()
