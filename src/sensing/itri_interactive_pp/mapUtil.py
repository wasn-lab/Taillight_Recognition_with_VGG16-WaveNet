#! ./sandbox/bin/python2.7
# coding=utf-8

import csv
import numpy as np
from PIL import Image, ImageDraw

def maskCrop(patch_box, mask_numpy):
    '''
    Crop map numpy mask by patchbox
    patch_box: (x_point, y_point, y_size, x_size)
    mask_numpy: binary numpy array of lanelet2 map drivable area
    '''
    # for itri_map
    map_x_min = 2008.593
    map_y_min = 41154.38
    
    # for zhubei_map
    # map_x_min = 633.018
    # map_y_min = 44853.908
    
    x = patch_box[0] - map_x_min
    y = patch_box[1] - map_y_min
    y_size = patch_box[2]
    x_size = patch_box[3]

    # solution*3
    x_pos = int(round(x) * 3)
    y_pos = int(round(y) * 3)
    
    if x_size <= 140:
        x_offset = 210
    else:
        x_offset = int(round(3 * x_size / 2))
    
    if y_size <= 140:
        y_offset = 210
    else:
        y_offset = int(round(3 * y_size / 2))
    # print("mask_crop x: ", x_pos)
    # print("mask_crop y: ", y_pos)

    # crop mask
    mask_list = []
    for i in range (y_pos - y_offset, y_pos + y_offset):
        # print("i: ", i)
        for j in range(x_pos - x_offset, x_pos + x_offset):
            # print("j: ", j)
            value = False
            try:
                value = mask_numpy[i, j]
            except:
                value = True
            # print(value)
            mask_list.append(value)
            
    cropped_mask = np.array(mask_list)
    cropped_mask = np.reshape(cropped_mask, (x_offset*2, y_offset*2))
    # print("mask: ", cropped_mask)
    # img = Image.new('1', (x_offset*2, y_offset*2))
    # pixels = img.load()
    # for i in range (img.size[0]):
    #     for j in range (img.size[1]):
    #         pixels[i, j] = cropped_mask[i][j],
    # save_png = str("map_mask/png/" + str(int(patch_box[0]*10)) + ".png")
    # img = img.save(save_png)

    return cropped_mask
            
if __name__ == '__main__':
    map_name = "itri_map"
    # map_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_mask/", map_name, ".npy")
    map_mask = np.load("map_mask/itri_map.npy")
    
    # x_min
    # y_min
    # for testing
    patch_box = (2008+233, 41172+221, 150, 150)
    map_mask = maskCrop(patch_box, map_mask)