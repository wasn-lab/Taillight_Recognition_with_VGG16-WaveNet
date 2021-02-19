#!/bin/sh
searchString1="jetson_clocks" 
searchString2="bash /home/nvidia/itriadv/src/scripts/bus_env/xavier/init_camera_driver_for_car_mode.sh"
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
            echo "init_camera_driver_for_car_mode.sh always in $file"
    else
            echo "#wait camera hardware ready"               | sudo tee -a $file
            echo "sleep 5"                          | sudo tee -a $file    
            echo "#init camera driver for car mode" | sudo tee -a $file
            echo "bash /home/nvidia/itriadv/src/scripts/bus_env/xavier/init_camera_driver_for_car_mode.sh" | sudo tee -a $file  
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
    echo "#init camera driver for car mode" | sudo tee -a $file
    echo "bash /home/nvidia/itriadv/src/scripts/bus_env/xavier/init_camera_driver_for_car_mode.sh" | sudo tee -a $file 
    sudo chmod +x /etc/rc.local   
fi

