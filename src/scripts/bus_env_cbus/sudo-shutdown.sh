#!/bin/bash
set -x

ssh -t xavier "echo nvidia | sudo -S shutdown now"
ssh -t local "echo itri | sudo -S shutdown now"
ssh -t camera "echo itri | sudo -S shutdown now"

echo All done
