import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import xml.etree.ElementTree as ET
tree = ET.parse('/home/mikelin/OpenRoadNet/Labeling/Labeled_Data/Label/MARCH_3_A2.xml')
root = tree.getroot()

for child in root.findall('image'):
    w = int(child.attrib['width'])
    h = int(child.attrib['height'])
    point_array = []
    color_ = 0
    im1 = np.zeros([h, w], dtype=np.int32)
    print(child.attrib['name'])
    name = child.attrib['name']
    tempx = 0.0
    for elem in child.findall('polygon'):
        # print(elem.attrib['label'])
        label = elem.attrib['label']
        if label == 'current' or label == 'Current':
            color_ = 1
        elif label == 'switchable' or label == 'Switchable':
            color_ = 2
        elif label == 'unswitchable'  or label == 'Unswitchable':
            color_ = 3

        point = elem.attrib['points']
        temp = ''
        temp_arr = []

        for s in range(len(point)):
            if point[s] != ',' and point[s] != ';':
                temp = temp + point[s]

            if point[s] == ',' or point[s] == ';' or s == len(point)-1:
                tempx = int(abs(float(temp)))
                temp = ''
            
            if point[s] == ',':
                temp_arr.append(tempx)
            elif point[s] == ';' or s == len(point)-1:
                temp_arr.append(tempx)
                point_array.append(temp_arr)
                temp_arr = []

        cv2.fillPoly( im1, np.int32([point_array]), color_ )
        # cv2.fillConvexPoly( im1, np.int32(point_array), 255 )
        point_array = []

    im_sv = Image.fromarray(im1).convert('RGB')
    im_sv.save('/home/mikelin/img_raw/Labeled/A2/April/_labeled_' + name)

plt.imshow(im1[...,::-1])
plt.show()