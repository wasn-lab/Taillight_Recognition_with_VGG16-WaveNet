if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    download_file(
        URL "http://nas.itriadv.co:8888/git_data/B1/camera_grabber/libgrabber_core_pc.so"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/libgrabber_core_pc.so"
        MD5 61f893c9a115d783ab20fe3a61ebb75f)
endif()

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
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/ac_22fps.ko"
        MD5 f4d5793ac0db45db637b4649ece6a8c3)
    download_file(
        URL "http://nas.itriadv.co:8888/git_data/B1/camera_grabber/jetson_xavier/init_ar0231"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/init_ar0231"
        MD5 6a3110150939e5fa33c9da8d319d036d)
    download_file(
        URL "http://nas.itriadv.co:8888/git_data/B1/camera_grabber/jetson_xavier/init_ar0231.sh"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/init_ar0231.sh"
        MD5 0224d8fa0c207e3a0205fecf05e3fdfe)
  endif()

endif()
