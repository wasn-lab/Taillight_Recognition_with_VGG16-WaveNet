if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    download_file(
        URL "http://118.163.54.109:8888/Share/ADV/S3_git_data/nvidia_camera_grabber/libgrabber_core_pc.so"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/libgrabber_core_pc.so"
        MD5 61f893c9a115d783ab20fe3a61ebb75f)
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    download_file(
        URL "http://118.163.54.109:8888/Share/ADV/S3_git_data/nvidia_camera_grabber/libgrabber_core.so"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/libgrabber_core.so"
        MD5 d27f025d5dc7d787cd7c75adb67c3f4f)
endif()