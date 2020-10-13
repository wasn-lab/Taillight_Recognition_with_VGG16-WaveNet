import numpy as np
import sys
import math
import os

Pi_UNIT = math.pi/180.0 

fx = 1235.85233
fy = 922.52899
ox = 636.74459
oy = 364.17232
c_alpha = 2.3 *Pi_UNIT #camera_pitch
c_beta = -0.1 *Pi_UNIT #camera_yaw
r_alpha = 0.34 *Pi_UNIT #radar_pitch
r_beta = -0.38 *Pi_UNIT #radar_yaw

Hcw = 0.86#camera_height (meter) 
Hr = 0.80#radar_height (meter) 
Ho = 0.80#object_height (meter)
Lx = 0 #(meter) 
Ly = 0 #(meter)


def radar_proj_xpyp(xr, zr):
    zrw = Ho/math.cos(r_alpha)
    xcw = xr*math.cos(r_beta) - ( zr*math.cos(r_alpha)+(zrw+Hr)*math.sin(r_alpha) )*math.sin(r_beta) - Lx
    ycw = xr*math.sin(r_beta) + ( zr*math.cos(r_alpha)+(zrw+Hr)*math.sin(r_alpha) )*math.cos(r_beta) + Ly
    zcw = -1*zr*math.sin(r_alpha) + Ho
    zc_new = (ycw*math.cos(c_alpha) - zcw*math.sin(c_alpha) + Hcw*math.sin(c_alpha))*math.cos(c_beta) - xcw*math.sin(c_beta)
    xc_new = ((ycw*math.cos(c_alpha) - zcw*math.sin(c_alpha) + Hcw*math.sin(c_alpha))*math.sin(c_beta) + xcw*math.cos(c_beta))
    yc_new = (-1*ycw*math.sin(c_alpha) - zcw*math.cos(c_alpha) + Hcw*math.cos(c_alpha))
    
    x_p = (xc_new/zc_new) * fx+ ox
    y_p = (yc_new/zc_new) * fy+ oy
    
    return x_p, y_p
    