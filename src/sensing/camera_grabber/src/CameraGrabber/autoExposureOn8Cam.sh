#!/bin/bash

function autoExposure() {
    # Enable Manual Exposure Settings
    ## 0xC8BC CAM_AET_AEMODE
    sudo i2ctransfer -f -y $1 w3@$2 0xC8 0xBC 0x06
    sleep 0.02

    ## 0xFC00 CMD_HANDLER_PARAMS_POOL0
    sudo i2ctransfer -f -y $1 w4@$2 0xFC 0x00 0x28 0x00
    sleep 0.5

    ## 0x0040 CMD_HANDLER_PARAMS_POOL32
    sudo i2ctransfer -f -y $1 w4@$2 0x00 0x40 0x81 0x00
    sleep 0.5
}

autoExposure 2 0x70
sleep 0.02
autoExposure 2 0x71
sleep 0.02
autoExposure 2 0x72
sleep 0.02
autoExposure 2 0x73
sleep 0.02
autoExposure 7 0x70
sleep 0.02
autoExposure 7 0x71
sleep 0.02
autoExposure 7 0x72
sleep 0.02
autoExposure 7 0x73
