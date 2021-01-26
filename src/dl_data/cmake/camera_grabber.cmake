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
        URL "http://nas.itriadv.co:8888/git_data/B1/camera_grabber/jetson_xavier/ar0231_22fps.ko"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/ar0231_22fps.ko"
        MD5 800ed0c012e8bdc778d951dcdb2118f6)
    download_file(
        URL "http://nas.itriadv.co:8888/git_data/B1/camera_grabber/jetson_xavier/init_ar0231_1207_9286trigger_v2.sh"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/init_ar0231_1207_9286trigger_v2.sh"
        MD5 29a3382f2f3b64dc28fdfd8444e8150b)    
  endif()

endif()
