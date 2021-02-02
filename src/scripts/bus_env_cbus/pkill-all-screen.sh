#!/bin/bash
set -x

pkill screen
ssh -t xavier "pill screen"
ssh -t local "pill screen"
ssh -t camera "pill screen"

echo All done
