#!/usr/bin/env python3

import collections, os, io, sys
import numpy as np

### For image distance estimation
### ===============================
Hor_regionHeightFC = [1207, 1120, 1044, 985, 940, 900, 868, 840, 818, 796, 780, 764, 748, 736, 724, 680, 648, 625, 608, 596, 584, 0]
Hor_regionDist = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50, 100]
Ver_regionHeightFC = [135, 463, 741, 990, 1262, 1543, 1806]
Ver_regionDist = [3, 2, 1, 0, -1, -2, -3]
Lidar_offset = 1
disEsti_imgHeight = 1208
disEsti_imgWidth = 1920
### ===============================
Current = np.zeros((1920,1208,2))

### Calculate cordinate in 3D from pixel
def GetPointDist(pixel_x, pixel_y, x_regionHeight, x_regionDist, y_regionHeight, y_regionDist, Lidar_offset):
    x_distMeter = 0
    y_distMeter = 0

    x_loc = pixel_y
    y_loc = pixel_x

    if len(x_regionDist) != 0:
        x_distMeter = ComputeObjectXDist(x_loc, x_regionHeight, x_regionDist)
    if len(y_regionDist) != 0:
        y_distMeter = ComputeObjectYDist(y_loc, x_loc, y_regionHeight, y_regionDist)

    return x_distMeter + Lidar_offset, y_distMeter

def ComputeObjectXDist(pixel_loc, regionHeight, regionDist):
    distance = -1
    unitLength = 0.0
    bias = 0
    offset = 0

    for i in range(1,len(regionHeight)):
        if pixel_loc >= regionHeight[i] and pixel_loc <= regionHeight[i-1]:
            regionpixel = regionHeight[i-1] - regionHeight[i]
            regionMeter = regionDist[i] - regionDist[i-1]
            unitLength = float(regionMeter) / float(regionpixel)
            bias = pixel_loc - regionHeight[i]
            offset = unitLength * float(bias)
            distance = regionDist[i] - offset

        elif pixel_loc <= regionHeight[i] and pixel_loc >= regionHeight[i-1]:
            regionpixel = regionHeight[i] - regionHeight[i-1]
            regionMeter = regionDist[i] - regionDist[i-1]
            unitLength = float(regionMeter)/float(regionpixel)
            bias = regionHeight[i] - pixel_loc
            offset = unitLength * float(bias)
            distance = regionDist[i] - offset
    
    multiplier = pow(10, 2)
    distance = int(distance * multiplier) / (multiplier*1.0)

    return distance

def ComputeObjectYDist(pixel_locY, pixel_locX, regionHeight, regionDist):
    distance = 0
    unitLength = 0.0
    bias = 0
    offset = 0.0
    # slope = [1.25, 1.66, 2.5, 5, 0, -5, -2.5, -1.66, -1.25]
    slope = [0.869, 1.45, 3.36, 0, -2.45, -1.3, -0.9]
    new_regionHeight = regionHeight.copy()

    for a in range(len(regionHeight)):
        _y = disEsti_imgHeight - pixel_locX
        if slope[a] != 0:
            new_regionHeight[a] = regionHeight[a] + int((1/slope[a]) * _y)
    
    for i in range(1,len(new_regionHeight)):
        if pixel_locY >= new_regionHeight[i] and pixel_locY <= new_regionHeight[i-1]:
            regionpixel = new_regionHeight[i-1] - new_regionHeight[i]
            regionMeter = regionDist[i] - regionDist[i-1]
            if regionpixel != 0:
                unitLength = float(regionMeter) / float(regionpixel)
            bias = pixel_locY - new_regionHeight[i]
            offset = unitLength * float(bias)
            distance = regionDist[i] - offset

        elif pixel_locY <= new_regionHeight[i] and pixel_locY >= new_regionHeight[i-1]:
            regionpixel = new_regionHeight[i] - new_regionHeight[i-1]
            regionMeter = regionDist[i] - regionDist[i-1]
            if regionpixel != 0:
                unitLength = float(regionMeter)/float(regionpixel)
            bias = new_regionHeight[i] - pixel_locY
            offset = unitLength * float(bias)
            distance = regionDist[i] - offset
        # Modifying: when pixel_locY < new_regionHeight[0] or > new_regionHeight[4] what will happen
        else:
            if pixel_locY < new_regionHeight[0]:
                distance = 5

            elif pixel_locY > new_regionHeight[len(new_regionHeight) - 1]:
                distance = -5
    
    multiplier = pow(10, 2)
    distance = int(distance * multiplier) / (multiplier*1.0)
    
    return distance

for y in range(1208):
    for x in range(1920):        
        a, b = GetPointDist(x, y, Hor_regionHeightFC, Hor_regionDist, Ver_regionHeightFC, Ver_regionDist, Lidar_offset)
        # print("[" + str(x) + "," + str(y) + "]:" + str(a) + "," + str(b))
        # Current.append([a, b])
        Current[x][y][0] = a
        Current[x][y][1] = b

print(Current)
print(Current.shape)

np.save('trans_mat_t.npy',Current)