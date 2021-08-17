#!/bin/bash

function manualExposure() {
    # $1 ISP Bus
    # $2 ISP Address
    # $3 $4 Exposure time in ms , The value is unsigned fixed-point with 7 fractional bits.
    # $5 $6 gain
    echo "ISP Bus $1"
    echo "ISP address $2"
    echo "Exposure time $3 $4"
    echo "Gain $5 $6"

    # Enable Manual Exposure Settings
    ## 0xC8BC CAM_AET_AEMODE
    sudo i2ctransfer -f -y $1 w3@$2 0xC8 0xBC 0x20
    #sleep 0.02
    sleep 1

    ## Make change config
    ## 0xFC00 CMD_HANDLER_PARAMS_POOL0
    sudo i2ctransfer -f -y $1 w4@$2 0xFC 0x00 0x28 0x00
    #sleep 0.5
    sleep 1

    ## 0x0040 CMD_HANDLER_PARAMS_POOL32
    sudo i2ctransfer -f -y $1 w4@$2 0x00 0x40 0x81 0x00
    #sleep 0.5
    sleep 1

    # Change Exposure Time
    ## 0xC8C0 CAM_AET_EXPOSURE_TIME_MS
    ## Pease modify it depending on your need
    sudo i2ctransfer -f -y $1 w4@$2 0xC8 0xC0 $3 $4
    #sleep 0.02
    sleep 1

    # Change Exposure Gain
    ## 0xC8C2 CAM_AET_EXPOSURE_GAIN
    ## Pease modify it depending on your need
    sudo i2ctransfer -f -y $1 w4@$2 0xC8 0xC2 $5 $6
    #sleep 0.02
    sleep 1

    ## 0xC842 CAM_SENSOR_CONTROL_EXPOSURE_REQUEST
    sudo i2ctransfer -f -y $1 w3@$2 0xC8 0x42 0x01
    #sleep 0.02
    sleep 1

    ## read cam exposure time
    #sudo i2ctransfer -f -y 2 w2@0x70 0xC8 0xC0 r2
    echo "Read exposure time"
    sudo i2ctransfer -f -y $1 w2@$2 0xC8 0xC0 r2
    sleep 1

    ## read cam exposure gain
    #sudo i2ctransfer -f -y 2 w2@0x70 0xC8 0xC2 r2
    echo "Read exposure gain"
    sudo i2ctransfer -f -y $1 w2@$2 0xC8 0xC2 r2
    sleep 1
    

    ## read cam 0 CAM_AET_AEMODE
    ## sudo i2ctransfer -f -y 2 w2@0x70 0xC8 0xBC r2
    echo "CAM_AET_AEMODE"
    sudo i2ctransfer -f -y $1 w2@$2 0xC8 0xBC r2
    sleep 1
    echo "---------------------"
}

manualExposure 2 0x70 $1 $2 $3 $4
sleep 0.02
manualExposure 2 0x71 $1 $2 $3 $4
sleep 0.02
manualExposure 2 0x72 $1 $2 $3 $4
sleep 0.02
manualExposure 2 0x73 $1 $2 $3 $4
sleep 0.02
manualExposure 7 0x70 $1 $2 $3 $4
sleep 0.02
manualExposure 7 0x71 $1 $2 $3 $4
sleep 0.02
manualExposure 7 0x72 $1 $2 $3 $4
sleep 0.02
manualExposure 7 0x73 $1 $2 $3 $4
