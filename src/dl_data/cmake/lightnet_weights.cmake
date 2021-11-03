download_file(
    URL "https://drive.google.com/u/0/uc?id=1jY9vaIUWL1EoGXjoRHYlADg2BqGKL4Lm&export=download"
    DESTINATION "${LIGHTNET_WEIGHTS_DIR}/iclu30_v3-kINT8-batch1.engine"
    MD5 c54f92402f16f4769b29d094ceea16c0
)

download_file(
   URL "https://drive.google.com/u/0/uc?id=160hpktrLzDhClipB9bvYJQcpdUHWLYUH&export=download"
    DESTINATION "${LIGHTNET_WEIGHTS_DIR}/iclu60_v3-kINT8-batch1.engine"
    MD5 074adae687675d17c6446718beaf074f
)

download_file(
   URL "drive.google.com/u/0/uc?id=1QgeuUWBXkYRHWRzacLLYZo0vjdHzhUwd&export=download"
    DESTINATION "${LIGHTNET_WEIGHTS_DIR}/libdetector.so"
    MD5 c3543102328474ae57ed3d18f38e69bd
)
