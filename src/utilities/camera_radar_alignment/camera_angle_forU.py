import numpy as np
import sys
import math
import os


if __name__=='__main__':

    #####camera_paramter####
    fx = 1235.85 
    fy = 922.53 

    ox = 636.745 
    oy = 364.172 
    sx = 13
    sy = 204
    h_camera = 1.139 #meter
    x_real = 0 #meter
    y_real = 5.6160 
    z_real = 0
    image_width = 1280 
    image_height = 720 
    ########################
    
    ####test angle ####
    test_vertical_down = -90.0
    test_vertical_up = 90.0
    test_horizontal_left = -90.0
    test_horizontal_right = 90.0
    ########################
    
    camera_alpha = float(2.0/180.0)*math.pi
    camera_beta = float(2.0/180.0)*math.pi
    test_line1 = oy + sy
    test_line1 = round(test_line1)
    test_line2 = ox + sx
    test_line2 = round(test_line2)
    test_line1 = int(test_line1)
    test_line2 = int(test_line2)
    x_c = 0.
    y_c = 0.
    z_c = 0.
    x_image = 0
    y_image = 0
    
    test_vertical_down = int(test_vertical_down * 10.0)
    test_vertical_up = int(test_vertical_up * 10.0 + 1.0)
    test_horizontal_left = int(test_horizontal_left * 10)
    test_horizontal_right = int(test_horizontal_right * 10.0 + 1.0)
    
    while(1):
        try:
            for i in range(test_vertical_down, test_vertical_up):
                for j in range(test_horizontal_left, test_horizontal_right):
                    k = float(i)
                    k = k/10.0
                    p = float(j)
                    p = p/10.0
                    
                    camera_alpha = float(k/180.0) * math.pi
                    camera_beta = float(p/180.0) * math.pi
                    x_c = x_real
                    y_c = -1 * y_real * math.sin(camera_alpha) - z_real * math.cos(camera_alpha) + h_camera * math.cos(camera_alpha)
                    z_c = y_real * math.cos(camera_alpha) - z_real * math.sin(camera_alpha) + h_camera * math.sin(camera_alpha)
                    
                    x_c_new = z_c * math.sin(camera_beta) + x_c * math.cos(camera_beta)
                    z_c_new = z_c * math.cos(camera_beta) - x_c * math.sin(camera_beta)
                    
                    r_image_x = (x_c_new / z_c_new) * fx + ox
                    r_image_y = (y_c / z_c_new) * fy + oy
                    
                    r_image_x = int(r_image_x)
                    r_image_y = int(r_image_y)
                    
                    if(abs(r_image_x - test_line2) < 2 and abs(r_image_y - test_line1) < 2):
                        print("V_angle:", k, ", H_angle:", p, ", r_image_x:", r_image_x, ", r_image_y:", r_image_y)
                    
                    if(r_image_x == test_line2 and r_image_y == test_line1):
                        print("V_angle_find:::::: ", k, ", H_angle_find: ", p, ", r_image_x: " , r_image_x, ", r_image_y: ", r_image_y)
                        break
                    
        except KeyboardInterrupt:
            os.system('pkill -9 python') 