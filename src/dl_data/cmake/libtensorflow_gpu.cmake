if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(LIBTENSORFLOW_TARGZ "libtensorflow-gpu-linux-x86_64-1.13.1.tar.gz")
    set(LIBTENSORFLOW_MD5 "d13656eb070e4284a48efa7dc9fcdd46")
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(LIBTENSORFLOW_TARGZ "libtensorflow-gpu-linux-aarch64-1.13.1.tar.gz")
    set(LIBTENSORFLOW_MD5 "c0498bef2989af00cffb497268f279e6")
endif()

download_file(
    URL "http://nas.itriadv.co:8888/git_data/B1/${LIBTENSORFLOW_TARGZ}"
    DESTINATION "${ASSETS_DIR}/${LIBTENSORFLOW_TARGZ}"
    MD5 ${LIBTENSORFLOW_MD5})

if(NOT EXISTS ${ASSETS_DIR}/lib/libtensorflow.so)
  message("unpacking ${LIBTENSORFLOW_TARGZ}")
  execute_process(COMMAND tar axf ${ASSETS_DIR}/${LIBTENSORFLOW_TARGZ}
                  WORKING_DIRECTORY ${ASSETS_DIR})
endif()
