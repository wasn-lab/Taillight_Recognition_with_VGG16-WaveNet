set(YOLOV3_NETWORK_CFG_FILE "${PARKNET_CFG_DIR}/parknet_yolov3.cfg")
set(YOLOV3_NETWORK_WEIGHTS_FILE "${PARKNET_WEIGHTS_DIR}/parknet-6k-1007.weight")
set(YOLOV3_OBJECT_NAMES_FILE "${PARKNET_CFG_DIR}/parknet.names")

set(TINY_YOLOV3_NETWORK_CFG_FILE "${PARKNET_CFG_DIR}/parknet_tiny_yolov3.cfg")
set(TINY_YOLOV3_NETWORK_WEIGHTS_FILE "${PARKNET_WEIGHTS_DIR}/parknet-tiny-yolov3-6k.weight")

set(NETWORK_CFG_FILE ${YOLOV3_NETWORK_CFG_FILE})
set(NETWORK_WEIGHTS_FILE ${YOLOV3_NETWORK_WEIGHTS_FILE})
set(OBJECT_NAMES_FILE ${YOLOV3_OBJECT_NAMES_FILE})


catkin_download_test_data(
    parknet_yolov3_weight
    "http://118.163.54.109:8888/Share/ADV/Hino_git_data/parknet/parknet-6k-1007.weight"
    DESTINATION ${PARKNET_WEIGHTS_DIR}
    MD5 bbb27b477914494942f68f7ad649b59f)

# experiment: use tiny yolo to speed up
catkin_download_test_data(
    parknet_tiny_yolov3_weight
    "http://118.163.54.109:8888/Share/ADV/Hino_git_data/parknet/parknet-tiny-yolov3-6k.weight"
    DESTINATION ${PARKNET_WEIGHTS_DIR}
    MD5 340c968505c38723e4f71e5e89835b74)

