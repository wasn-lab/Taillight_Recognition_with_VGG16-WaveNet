if (NOT ${ENABLE_CAMERA_GSTREAMER_GRABBER})
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    download_file(
        URL "http://nas.itriadv.co:8888/git_data/B1/camera_grabber/libgrabber_core.so"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/libgrabber_core.so"
        MD5 d27f025d5dc7d787cd7c75adb67c3f4f)
  endif()
else()
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    download_file(
        URL "http://nas.itriadv.co:8888/git_data/B1/camera_grabber/jetson_xavier/ac_22fps.ko"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/ar0231.ko"
        MD5 c5e84a49fa0785b2976e7a3798e318b2)
    download_file(
        URL "http://nas.itriadv.co:8888/git_data/B1/camera_grabber/jetson_xavier/init_ar0231"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/init_ar0231.sh"
        MD5 43368e927bb809d8665cd5ea76d81311)    
  endif()

endif()
