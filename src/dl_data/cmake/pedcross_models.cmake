download_file(
    URL "http://nas.itriadv.co:8888/git_data/B1/ped_models/rf_10frames_normalization_15peek.yml"
    DESTINATION "${PEDCROSS_MODELS_DIR}/rf_10frames_normalization_15peek.yml"
    MD5 c0ec26c803a7d5bcb0b63edb6509bcd4)

download_file(
    URL "http://nas.itriadv.co:8888/git_data/B1/ped_models/pose/body_25/pose_iter_584000.caffemodel"
    DESTINATION "${PEDCROSS_MODELS_DIR}/pose/body_25/pose_iter_584000.caffemodel"
    MD5 78287b57cf85fa89c03f1393d368e5b7)

download_file(
    URL "http://nas.itriadv.co:8888/git_data/B1/ped_models/pose/body_25/pose_deploy.prototxt"
    DESTINATION "${PEDCROSS_MODELS_DIR}/pose/body_25/pose_deploy.prototxt"
    MD5 451938635cf007327c3cf235ed483844)
