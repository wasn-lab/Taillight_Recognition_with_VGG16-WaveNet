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
        URL "http://nas.itriadv.co:8888/git_data/B1/camera_grabber/jetson_xavier/init_ar0231_20210331v3_2.sh"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/init_ar0231_20210331v3_2.sh"
        MD5 7902f7370b306a9949dd83dedbfe613d)
    execute_process(COMMAND chmod +x ${CAMERA_GRABBER_DATA_DIR}/init_ar0231_20210331v3_2.sh)
    download_file(
        URL "http://nas.itriadv.co:8888/git_data/B1/camera_grabber/jetson_xavier/libmvextractor_standalone.so.1.0.1"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/libmvextractor_standalone.so.1.0.1"
        MD5 8767bd11064871dcc838f4d756d780e1)        
  endif()

endif()
