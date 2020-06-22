download_file(
    URL "http://118.163.54.109:8888/Share/ADV/S3_git_data/drivenet/yolov3_b1.weights"
    DESTINATION "${DRIVENET_WEIGHTS_DIR}/yolov3_b1.weights"
    MD5 aa741933490b2ac57e03cae551857744)

download_file(
    URL "http://118.163.54.109:8888/Share/ADV/S3_git_data/drivenet/yolov3_fov120_b1.weights"
    DESTINATION "${DRIVENET_WEIGHTS_DIR}/yolov3_fov120_b1.weights"
    MD5 a30ed70c06a7439accdb66b05efad392)
