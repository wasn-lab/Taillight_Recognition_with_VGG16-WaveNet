#!/bin/sh
searchString1="jetson_clocks" 
searchString2="bash /home/nvidia/itriadv/src/sensing/camera_grabber/src/CameraGrabber/init_ar0231_driver_c2.sh /home/nvidia/itriadv/src/sensing/camera_grabber/src/CameraGrabber nvidia false"
searchString3="bash /home/nvidia/itriadv/src/sensing/camera_grabber/src/CameraGrabber/exposure-setup-day-or-night.sh /home/nvidia/itriadv/src/sensing/camera_grabber/src/CameraGrabber"
file="/etc/rc.local"
if [ -f "/etc/rc.local" ]; then    
    echo "File /etc/rc.local exists. Append commands to the end"
    if grep -Fxq "$searchString1" $file
    then
            echo "jetson_clocks always in $file"
    else
            echo "#init max clock"                  | sudo tee -a $file
            echo "sleep 5"                          | sudo tee -a $file
            echo "jetson_clocks"                    | sudo tee -a $file
    fi
    if grep -Fxq "$searchString2" $file
    then
            echo "init_ar0231_driver_c2.sh always in $file"
    else
            echo "#wait camera hardware ready"               | sudo tee -a $file
            echo "sleep 5"                          | sudo tee -a $file    
            echo "#init camera driver for laoratory mode, setup EQ and Pre-Emp" | sudo tee -a $file
            echo "# parameter 1 : the path of camera driver" | sudo tee -a $file
            echo "# parameter 2 : the root password" | sudo tee -a $file
            echo "# parameter 3 : True mean car mode, False mean laboratory mode" | sudo tee -a $file
            echo "bash /home/nvidia/itriadv/src/sensing/camera_grabber/src/CameraGrabber/init_ar0231_driver_c2.sh /home/nvidia/itriadv/src/sensing/camera_grabber/src/CameraGrabber nvidia false" | sudo tee -a $file  
    fi
    if grep -Fxq "$searchString3" $file
    then
            echo "exposure-setup-day-or-night.sh always in $file"
    else               
            echo "#init camera exposure" | sudo tee -a $file
            echo "# parameter 1 : the path of camera exposure script" | sudo tee -a $file
            echo "bash /home/nvidia/itriadv/src/sensing/camera_grabber/src/CameraGrabber/exposure-setup-day-or-night.sh /home/nvidia/itriadv/src/sensing/camera_grabber/src/CameraGrabber" | sudo tee -a $file  
    fi
    sudo chmod +x /etc/rc.local
else
    echo "File /etc/rc.local does not exists. create new $file"
    echo "#!/bin/sh -e"                     | sudo tee -a $file
    echo "exec > /tmp/rclocal.log 2>&1"     | sudo tee -a $file
    echo "set -x"                           | sudo tee -a $file
    echo "#init max clock"                  | sudo tee -a $file
    echo "sleep 5"                          | sudo tee -a $file
    echo "jetson_clocks"                    | sudo tee -a $file
    echo "#wait camera hardware ready"               | sudo tee -a $file
    echo "sleep 5"                          | sudo tee -a $file    
    echo "#init camera driver for laoratory mode, setup EQ and Pre-Emp" | sudo tee -a $file
    echo "# parameter 1 : the path of camera driver" | sudo tee -a $file
    echo "# parameter 2 : the root password" | sudo tee -a $file
    echo "# parameter 3 : True mean car mode, False mean laboratory mode" | sudo tee -a $file
    echo "bash /home/nvidia/itriadv/src/sensing/camera_grabber/src/CameraGrabber/init_ar0231_driver_c2.sh /home/nvidia/itriadv/src/sensing/camera_grabber/src/CameraGrabber nvidia false" | sudo tee -a $file
    echo "#init camera exposure" | sudo tee -a $file
    echo "# parameter 1 : the path of camera exposure script" | sudo tee -a $file
    echo "bash /home/nvidia/itriadv/src/sensing/camera_grabber/src/CameraGrabber/exposure-setup-day-or-night.sh /home/nvidia/itriadv/src/sensing/camera_grabber/src/CameraGrabber" | sudo tee -a $file  
    sudo chmod +x /etc/rc.local   
fi

