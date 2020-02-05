download_file(
    URL "http://118.163.54.109:8888/Share/ADV/S3_git_data/ped_models/rf_5frames_normalization_15peek.yml"
    DESTINATION "${PEDCROSS_MODELS_DIR}/rf_5frames_normalization_15peek.yml"
    MD5 7476d1ba785b806255c7df62e0984249)

download_file(
    URL "http://118.163.54.109:8888/Share/ADV/S3_git_data/ped_models/rf_10frames_normalization_15peek.yml"
    DESTINATION "${PEDCROSS_MODELS_DIR}/rf_10frames_normalization_15peek.yml"
    MD5 c0ec26c803a7d5bcb0b63edb6509bcd4)

download_file(
    URL "http://118.163.54.109:8888/Share/ADV/S3_git_data/ped_models/body_25/pose_iter_584000.caffemodel"
    DESTINATION "${PEDCROSS_MODELS_DIR}/pose/body_25/pose_iter_584000.caffemodel"
    MD5 78287b57cf85fa89c03f1393d368e5b7)

download_file(
    URL "http://118.163.54.109:8888/Share/ADV/S3_git_data/ped_models/body_25/pose_deploy.prototxt"
    DESTINATION "${PEDCROSS_MODELS_DIR}/pose/body_25/pose_deploy.prototxt"
    MD5 451938635cf007327c3cf235ed483844)
