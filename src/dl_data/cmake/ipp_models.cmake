download_file(
    URL "http://nas.itriadv.co:8888/git_data/B1/ipp_models/nuScene_config.json"
    DESTINATION "${IPP_MODELS_DIR}/int_ee/config.json"
    MD5 161343a2fcfa76d41522346a78ab01a5)

download_file(
    URL "http://nas.itriadv.co:8888/git_data/B1/ipp_models/nuScene_model_registrar-12.pt"
    DESTINATION "${IPP_MODELS_DIR}/int_ee/model_registrar-12.pt"
    MD5 4431bc85e9c2d9db2c7f77d42c3f2f4a)
