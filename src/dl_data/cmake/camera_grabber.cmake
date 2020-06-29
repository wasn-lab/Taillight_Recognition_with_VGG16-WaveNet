if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    download_file(
        URL "http://nas.itriadv.co:8888/git_data/B1/camera_grabber/libgrabber_core_pc.so"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/libgrabber_core_pc.so"
        MD5 61f893c9a115d783ab20fe3a61ebb75f)
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    download_file(
        URL "http://nas.itriadv.co:8888/git_data/B1/camera_grabber/libgrabber_core.so"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/libgrabber_core.so"
        MD5 d27f025d5dc7d787cd7c75adb67c3f4f)
endif()