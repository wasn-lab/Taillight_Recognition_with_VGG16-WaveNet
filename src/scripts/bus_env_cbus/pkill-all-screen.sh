#!/bin/bash
set -x

pkill screen
ssh -t xavier "pkill screen"
ssh -t local "pkill screen"
ssh -t camera "pkill screen"
ssh -t throttle "pkill screen"

echo All done
