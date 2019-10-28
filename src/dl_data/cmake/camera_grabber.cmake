if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    download_file(
        URL "https://onedrive.live.com/download?cid=FE58073B2C9B12D7&resid=FE58073B2C9B12D7%212471&authkey=AIVTzrciseMlEH0"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/libgrabber_core_pc.so"
        MD5 61f893c9a115d783ab20fe3a61ebb75f)
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    download_file(
        URL "https://onedrive.live.com/download?cid=FE58073B2C9B12D7&resid=FE58073B2C9B12D7%212470&authkey=ANeHk1KZXQqw2dE"
        DESTINATION "${CAMERA_GRABBER_DATA_DIR}/libgrabber_core.so"
        MD5 d27f025d5dc7d787cd7c75adb67c3f4f)
endif()
