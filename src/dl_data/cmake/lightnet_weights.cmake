download_file(
    URL "140.96.172.66:8080/share.cgi?ssid=0zcxNW1&fid=0zcxNW1&filename=iclu30_v3-kINT8-batch1.engine&openfolder=forcedownload&ep="
    DESTINATION "${LIGHTNET_WEIGHTS_DIR}/iclu30_v3-kINT8-batch1.engine"
    MD5 c54f92402f16f4769b29d094ceea16c0
)

download_file(
   URL "140.96.172.66:8080/share.cgi?ssid=0fDnRP7&fid=0fDnRP7&filename=iclu60_v3-kINT8-batch1.engine&openfolder=forcedownload&ep="
    DESTINATION "${LIGHTNET_WEIGHTS_DIR}/iclu60_v3-kINT8-batch1.engine"
    MD5 074adae687675d17c6446718beaf074f
)

download_file(
   URL "140.96.172.66:8080/share.cgi?ssid=0RTQaYe&fid=0RTQaYe&filename=libdetector.so&openfolder=forcedownload&ep="
    DESTINATION "${LIGHTNET_WEIGHTS_DIR}/libdetector.so"
    MD5 c3543102328474ae57ed3d18f38e69bd
)
