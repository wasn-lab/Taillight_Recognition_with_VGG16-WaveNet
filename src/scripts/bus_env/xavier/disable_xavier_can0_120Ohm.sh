echo 332 > /sys/class/gpio/export #CAN0_TERM
echo out > /sys/class/gpio/gpio332/direction
echo 0 > /sys/class/gpio/gpio332/value
