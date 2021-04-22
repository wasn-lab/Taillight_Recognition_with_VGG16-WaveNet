#!/bin/bash
set -x

pkill screen
ssh -t xavier "pkill screen"
ssh -t local "pkill screen"
ssh -t camera "pkill screen"
xdotool windowkill `wmctrl -l | grep 駕駛艙畫面 | cut -f 1 -d " "`

echo All done
