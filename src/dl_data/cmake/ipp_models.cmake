download_file(
    URL "http://nas.itriadv.co:8888/git_data/B1/ipp_models/nuScene_config.json"
    DESTINATION "${IPP_MODELS_DIR}/int_ee/config.json"
    MD5 161343a2fcfa76d41522346a78ab01a5)

download_file(
    URL "http://nas.itriadv.co:8888/git_data/B1/ipp_models/nuScene_model_registrar-12.pt"
    DESTINATION "${IPP_MODELS_DIR}/int_ee/model_registrar-12.pt"
    MD5 4431bc85e9c2d9db2c7f77d42c3f2f4a)

download_file(
    URL "http://nas.itriadv.co:8888/git_data/B1/ipp_models/nuScene_model_with_map/config.json"
    DESTINATION "${IPP_MODELS_DIR}/int_ee_me/config.json"
    MD5 ebc3b7cf216aeac086913ff9dacc7f61)

download_file(
    URL "http://nas.itriadv.co:8888//git_data/B1/ipp_models/nuScene_model_with_map/model_registrar-12.pt"
    DESTINATION "${IPP_MODELS_DIR}/int_ee_me/model_registrar-12.pt"
    MD5 92d8921e0d7f2ded5ab65c69c84c874e)

download_file(
    URL "http://nas.itriadv.co:8888/git_data/B1/ipp_models/map_mask/ITRI_lanelet2_map.npy"
    DESTINATION "${IPP_MAP_MASKS_DIR}/ITRI_lanelet2_map.npy"
    MD5 98327890852f05bbc7a7672d79f13fc1)

download_file(
    URL "http://nas.itriadv.co:8888//git_data/B1/ipp_models/map_mask/Zhubei_lanelet2_map.npy"
    DESTINATION "${IPP_MAP_MASKS_DIR}/Zhubei_lanelet2_map.npy"
    MD5 997dc6151b787c4e8487a6af3116f79b)