import numpy as np
import sys
import math
import os


if __name__=='__main__':
    
    Pi_UNIT = math.pi/180.0

    #####camera_paramter####
    fx = 1235.85 
    fy = 922.53 
    ox = 636.745 
    oy = 364.172     
    Hcw = 1.139 #meter
    image_width = 1280 
    image_height = 720     
    c_alpha = -1.0 *Pi_UNIT 
    c_beta = 0.6 *Pi_UNIT 
    ########################
    
    #####radar_paramter####
    Hr = 1.080 #meter
    Ho = 1.080 
    Lx = 0 #meter
    Ly = 0 
    xp = 655
    yp = 382
    angle = -0.2
    dist = 5.7
    xr = dist*math.sin(angle*Pi_UNIT)
    yr = dist*math.cos(angle*Pi_UNIT)
    ########################
    
    ####test angle ####
    test_vertical_down = -30.0
    test_vertical_up = 30.0
    test_horizontal_left = -30.0
    test_horizontal_right = 30.0
    
    test_vertical_down = int(test_vertical_down * 100.0)
    test_vertical_up = int(test_vertical_up * 100.0 + 1.0)
    test_horizontal_left = int(test_horizontal_left * 100.0)
    test_horizontal_right = int(test_horizontal_right * 100.0 + 1.0)
    ########################
    print("---------------------")
    while(1):
        try:
            for i in range(test_vertical_down, test_vertical_up):
                for j in range(test_horizontal_left, test_horizontal_right):
                    k = float(i)
                    k = k/100.0
                    
                    p = float(j)
                    p = p/100.0
                    
                    radar_alpha = float(k/180.0) * math.pi
                    radar_beta = float(p/180.0) * math.pi

                    Xcw = xr*math.cos(radar_beta) - yr*math.cos(radar_alpha)*math.sin(radar_beta) - (Ho/math.cos(radar_alpha) + Hr)*math.sin(radar_alpha)*math.sin(radar_beta) - Lx
                    Ycw = xr*math.sin(radar_beta) + yr*math.cos(radar_alpha)*math.cos(radar_beta) + (Ho/math.cos(radar_alpha) + Hr)*math.sin(radar_alpha)*math.cos(radar_beta) + Ly
                    Zcw = -yr*math.sin(radar_alpha) + Ho
                    
                    Zc_new = (Ycw*math.cos(c_alpha) - Zcw*math.sin(c_alpha) + Hcw*math.sin(c_alpha))*math.cos(c_beta) - Xcw*math.sin(c_beta)
                    Xc_new = ((Ycw*math.cos(c_alpha) - Zcw*math.sin(c_alpha) + Hcw*math.sin(c_alpha))*math.sin(c_beta) + Xcw*math.cos(c_beta))
                    Yc_new = (-1*Ycw*math.sin(c_alpha) - Zcw*math.cos(c_alpha) + Hcw*math.cos(c_alpha))
                    
                    x_p_hat = (Xc_new/Zc_new) * fx+ ox
                    y_p_hat = (Yc_new/Zc_new) * fy+ oy
                    

                    if(abs(x_p_hat - xp) < 2 and abs(y_p_hat - yp) < 2):
                        print("V_angle:", k, ", H_angle:", p, ", x_p_hat = ", x_p_hat, ", y_p_hat = ", y_p_hat)
                            
                    if(int(x_p_hat) == int(xp) and int(y_p_hat) == int(yp)):
                        print("V_angle_find:::::::::: ", k, ", H_angle_find: ", p, ", x_p_hat: " , x_p_hat, ", y_p_hat: ", y_p_hat)
                        break
                        
        except KeyboardInterrupt:
            os.system('pkill -9 python')   