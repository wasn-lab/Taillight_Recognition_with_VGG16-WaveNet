# data analysis for LiDAR labeling file
import sys
import glob
import os
import math
import string
import rospy
from tabulate import tabulate
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

def input_parser(inputpath):
    objectArray = []
    tree = ET.ElementTree(file=inputpath)
    if tree.find("lidar/frame_shadow").text != "1":
        for elem in tree.iterfind('lidar/object'):
            id = int(elem[0].text)  # ID
            objectType = elem[1].text   # type
            xCenter = float(elem.find('pcd_bndbox/x_center').text) # pcd_bndbox/x_center
            yCenter = float(elem.find('pcd_bndbox/y_center').text) # pcd_bndbox/y_center
            zCenter = float(elem.find('pcd_bndbox/z_center').text) # pcd_bndbox/z_center
            shadow = int(elem.find('pcd_bndbox/shadow').text) # pcd_bndbox/shadow
            objectArray.append([id, objectType, xCenter, yCenter, zCenter, shadow])
        return objectArray
    else:
        print("Frame has shadow")
        for elem in tree.iterfind('lidar/object'):
            id = int(elem[0].text)  # ID
            objectType = elem[1].text   # type
            xCenter = float(elem.find('pcd_bndbox/x_center').text) # pcd_bndbox/x_center
            yCenter = float(elem.find('pcd_bndbox/y_center').text) # pcd_bndbox/y_center
            zCenter = float(elem.find('pcd_bndbox/z_center').text) # pcd_bndbox/z_center
            shadow = int(elem.find('pcd_bndbox/shadow').text) # pcd_bndbox/shadow
            objectArray.append([id, objectType, xCenter, yCenter, zCenter, shadow])
        return objectArray

def object_counter(objectArray):
    # count variable
    # car, ped, bike, front, 
    # front_right, front_left, right, left,
    # rear_right, rear_left, rear
    cnt_list = [0 for x in range(16)]
    cnt_detail_list = [0 for x in range(120)]
    count = 0
    shadow_count = 0
    a = 0
    b = 0
    c = 0
    for item in objectArray:
        if item[-1] == 0:
            count = count + 1
            if item[1] == "Car":
                cnt_list[0] = cnt_list[0] + 1
                b = 0
            elif item[1] == "Pedestrian":
                cnt_list[1] = cnt_list[1] + 1
                b = 1
            elif item[1] == "Bicycle" or item[1] == "Motorcycle" or item[1] == "Motorbike":
                cnt_list[2] = cnt_list[2] + 1
                b = 2
            elif  item[1] == "Bus":
                cnt_list[3] = cnt_list[3] + 1
                b = 3
            elif item[1] == "Truck":
                cnt_list[4] = cnt_list[4] +1
                b = 4
            else:
                print (item[1])
            
            if item[2] != 0:
                # front or rear
                if abs(item[3]/item[2]) <= float(1)/float(1+math.sqrt(2)):
                    # front
                    if item[2] > 0:
                        cnt_list[5] = cnt_list[5] + 1
                        c = 0
                    # rear
                    else:
                        cnt_list[12] = cnt_list[12] + 1
                        c = 7
                # right or left
                elif abs(item[3]/item[2]) >= float(1+math.sqrt(2)):
                    # right
                    if item[3] < 0:
                        cnt_list[8] = cnt_list[8] + 1
                        c = 3
                    # left
                    else:
                        cnt_list[9] = cnt_list[9] + 1
                        c = 4
                # others
                else:
                    # front_right
                    if item[2] > 0 and item[3] < 0:
                        cnt_list[6] = cnt_list[6] + 1
                        c = 1
                    # front_left
                    elif item[2] > 0 and item[3] > 0:
                        cnt_list[7] = cnt_list[7] + 1
                        c = 2
                    # rear_right
                    elif item[2] < 0 and item[3] < 0:
                        cnt_list[10] = cnt_list[10] + 1
                        c = 5
                    # rear_left
                    elif item[2] < 0 and item[3] > 0:
                        cnt_list[11] = cnt_list[11] + 1
                        c = 6
            else:
                if item[3] < 0:
                    cnt_list[8] = cnt_list[8] + 1
                    c = 3
                elif item[3] > 0:
                    cnt_list[9] = cnt_list[9] + 1
                    c = 4
            
            dist_square = math.pow(item[2], 2) + math.pow(item[3], 2)
            if dist_square <= 225:
                cnt_list[13] = cnt_list[13] + 1
                a = 0
            elif dist_square > 900:
                cnt_list[15] = cnt_list[15] + 1
                a = 2
            else:
                cnt_list[14] = cnt_list[14] + 1
                a = 1
            
            cnt_detail_list[a*40+b*8+c] = cnt_detail_list[a*40+b*8+c] + 1
        else:
            shadow_count = shadow_count + 1
            count = count + 1
        
    return cnt_list, cnt_detail_list, count, shadow_count

def main():
    total_obj_cnts = [0 for x in range(16)]
    total_obj_detail_cnts = [0 for x in range(120)]
    total_shadow_cnts = 0
    total_total_cnts = 0
    total_file_num = 0

    extracted_dir_list = ["Dummy"]
    print ("** Start Analyzing **")
    current_dir_path = os.path.dirname(os.path.abspath('__file__'))
    os.chdir(current_dir_path)
    root_dir = current_dir_path
    for dir_name, sub_dir_list, file_list in os.walk(root_dir):
        for fname in file_list:
            if fname.endswith(".xml"):
                if dir_name in extracted_dir_list:
                    pass
                else:
                    extracted_dir_list.append(dir_name)
                    os.chdir(dir_name)

                    current_dir = os.getcwd()
                    print ("Current Directory: %s" % current_dir)
                    print ("File in Directory: %s" % current_dir)
                    if os.path.isdir(current_dir) and not (dir_name.startswith(".")):
                        xml_list = sorted(os.listdir(os.getcwd()))
                        # number of xml files in the directory
                        dir_file_num = len(xml_list)
                        # statistics of objects in xml files in the directory
                        dir_obj_cnts = [0 for x in range(16)]
                        # details of statistics of objects in xml files in the directory
                        dir_obj_detail_cnts = [0 for x in range(120)]
                        # number of objects in the directory
                        total_cnts = 0
                        dir_shadow_cnts = 0
                        for xml_file in xml_list:
                            if xml_file.endswith(".xml"):
                                print ("Current File:%s" % xml_file)
                                file_name = xml_file.split("/")[-1].split(".")[0]
                                print (file_name)

                                xml_path = current_dir + "/" + xml_file
                                xml_objs = input_parser(xml_path)
                                if xml_objs is not None:
                                    obj_cnts, obj_detail_cnts, file_cnts, shadow_cnts = object_counter(xml_objs)
                                    dir_obj_cnts = [dir_obj_cnts[i] + obj_cnts[i] for i in range(len(obj_cnts))]
                                    dir_obj_detail_cnts = [dir_obj_detail_cnts[i] + obj_detail_cnts[i] for i in range(len(obj_detail_cnts))]
                                    dir_shadow_cnts = dir_shadow_cnts + shadow_cnts
                                    total_cnts = total_cnts + file_cnts
                        # print statistics
                        print ("** Statistics of %s **" % dir_name )
                        print ("Total Files: %d" % dir_file_num)
                        print ("Total Objects: %d" % total_cnts)
                        print ("Shadowed Objectes: %d" % dir_shadow_cnts)
                        print ("Classified by Type:")
                        print ("    Objects of Type Car: %d" % dir_obj_cnts[0])
                        print ("    Objects of Type Pedestrian: %d" % dir_obj_cnts[1])
                        print ("    Objects of Type Cyclist: %d" % dir_obj_cnts[2])
                        print ("    Objects of Type Bus: %d" % dir_obj_cnts[3])
                        print ("    Objects of Type Truck: %d" % dir_obj_cnts[4])
                        print ("Classified by direction:")
                        print ("    Objects of in front: %d" % dir_obj_cnts[5])
                        print ("    Objects of in front right: %d" % dir_obj_cnts[6])
                        print ("    Objects of in front left: %d" % dir_obj_cnts[7])
                        print ("    Objects of in right: %d" % dir_obj_cnts[8])
                        print ("    Objects of in left: %d" % dir_obj_cnts[9])
                        print ("    Objects of in rear right: %d" % dir_obj_cnts[10])
                        print ("    Objects of in rear left: %d" % dir_obj_cnts[11])
                        print ("    Objects of in rear: %d" % dir_obj_cnts[12])
                        print ("Classified by range:")
                        print ("    Objects in 15m: %d" % dir_obj_cnts[13])
                        print ("    Objects between 15m to 30m: %d" % dir_obj_cnts[14])
                        print ("    Objects over 30m: %d" % dir_obj_cnts[15])

                        print ("\n** Details of %s **" % dir_name)
                        headers = ["Car", "Pedestrian", "Cyclist", "Bus", "Truck"]
                        table_elem_num = 40
                        table = []
                        for i in range(3):
                            table.append([["front", dir_obj_detail_cnts[0+i*table_elem_num], dir_obj_detail_cnts[8+i*table_elem_num], dir_obj_detail_cnts[16+i*table_elem_num], dir_obj_detail_cnts[24+i*table_elem_num], dir_obj_detail_cnts[32+i*table_elem_num]],
                                                        ["front_right", dir_obj_detail_cnts[1+i*table_elem_num], dir_obj_detail_cnts[9+i*table_elem_num], dir_obj_detail_cnts[17+i*table_elem_num], dir_obj_detail_cnts[25+i*table_elem_num], dir_obj_detail_cnts[33+i*table_elem_num]],
                                                        ["front_left", dir_obj_detail_cnts[2+i*table_elem_num], dir_obj_detail_cnts[10+i*table_elem_num], dir_obj_detail_cnts[18+i*table_elem_num], dir_obj_detail_cnts[26+i*table_elem_num], dir_obj_detail_cnts[34+i*table_elem_num]],
                                                        ["right", dir_obj_detail_cnts[3+i*table_elem_num], dir_obj_detail_cnts[11+i*table_elem_num], dir_obj_detail_cnts[19+i*table_elem_num], dir_obj_detail_cnts[27+i*table_elem_num], dir_obj_detail_cnts[35+i*table_elem_num]],
                                                        ["left", dir_obj_detail_cnts[4+i*table_elem_num], dir_obj_detail_cnts[12+i*table_elem_num], dir_obj_detail_cnts[20+i*table_elem_num], dir_obj_detail_cnts[28+i*table_elem_num], dir_obj_detail_cnts[36+i*table_elem_num]],
                                                        ["rear_right", dir_obj_detail_cnts[5+i*table_elem_num], dir_obj_detail_cnts[13+i*table_elem_num], dir_obj_detail_cnts[21+i*table_elem_num], dir_obj_detail_cnts[29+i*table_elem_num], dir_obj_detail_cnts[37+i*table_elem_num]],
                                                        ["rear_left", dir_obj_detail_cnts[6+i*table_elem_num], dir_obj_detail_cnts[14+i*table_elem_num], dir_obj_detail_cnts[22+i*table_elem_num], dir_obj_detail_cnts[30+i*table_elem_num], dir_obj_detail_cnts[38+i*table_elem_num]],
                                                        ["rear", dir_obj_detail_cnts[7+i*table_elem_num], dir_obj_detail_cnts[15+i*table_elem_num], dir_obj_detail_cnts[23+i*table_elem_num], dir_obj_detail_cnts[31+i*table_elem_num], dir_obj_detail_cnts[39+i*table_elem_num]]])
                            print (tabulate(table[i], headers, tablefmt="pretty", numalign="right", colalign=("center", "center", "center", "center", "center")))
                        
                        total_obj_cnts = [total_obj_cnts[i] + dir_obj_cnts[i] for i in range(len(total_obj_cnts))]
                        total_obj_detail_cnts = [total_obj_detail_cnts[i] + dir_obj_detail_cnts[i] for i  in range(len(total_obj_detail_cnts))]
                        total_total_cnts = total_total_cnts + total_cnts
                        total_shadow_cnts = total_shadow_cnts + dir_shadow_cnts
                        total_file_num = total_file_num + dir_file_num
    
    # print statistics
    print ("** Statistics in total **")
    print ("Total Objects: %d" % total_total_cnts)
    print ("Total Files: %d" % total_file_num)
    print ("Shadowed Objectes: %d" % total_shadow_cnts)
    print ("Classified by Type:")
    print ("    Objects of Type Car: %d" % total_obj_cnts[0])
    print ("    Objects of Type Pedestrian: %d" % total_obj_cnts[1])
    print ("    Objects of Type Cyclist: %d" % total_obj_cnts[2])
    print ("    Objects of Type Bus: %d" % total_obj_cnts[3])
    print ("    Objects of Type Truck: %d" % total_obj_cnts[4])
    print ("Classified by direction:")
    print ("    Objects of in front: %d" % total_obj_cnts[5])
    print ("    Objects of in front right: %d" % total_obj_cnts[6])
    print ("    Objects of in front left: %d" % total_obj_cnts[7])
    print ("    Objects of in right: %d" % total_obj_cnts[8])
    print ("    Objects of in left: %d" % total_obj_cnts[9])
    print ("    Objects of in rear right: %d" % total_obj_cnts[10])
    print ("    Objects of in rear left: %d" % total_obj_cnts[11])
    print ("    Objects of in rear: %d" % total_obj_cnts[12])
    print ("Classified by range:")
    print ("    Objects in 15m: %d" % total_obj_cnts[13])
    print ("    Objects between 15m to 30m: %d" % total_obj_cnts[14])
    print ("    Objects over 30m: %d" % total_obj_cnts[15])

    print ("\n** Details in total **")
    headers = ["Car", "Pedestrian", "Cyclist", "Bus", "Truck"]
    table_elem_num = 40
    table = []
    for i in range(3):
        if i == 0:
            print("\n to 15m")
        elif i == 1:
            print("\n from 15m to 30m")
        else:
            print("\n further than 30m")
        table.append([["front", total_obj_detail_cnts[0+i*table_elem_num], total_obj_detail_cnts[8+i*table_elem_num], total_obj_detail_cnts[16+i*table_elem_num], total_obj_detail_cnts[24+i*table_elem_num], total_obj_detail_cnts[32+i*table_elem_num]],
                                    ["front_right", total_obj_detail_cnts[1+i*table_elem_num], total_obj_detail_cnts[9+i*table_elem_num], total_obj_detail_cnts[17+i*table_elem_num], total_obj_detail_cnts[25+i*table_elem_num], total_obj_detail_cnts[33+i*table_elem_num]],
                                    ["front_left", total_obj_detail_cnts[2+i*table_elem_num], total_obj_detail_cnts[10+i*table_elem_num], total_obj_detail_cnts[18+i*table_elem_num], total_obj_detail_cnts[26+i*table_elem_num], total_obj_detail_cnts[34+i*table_elem_num]],
                                    ["right", total_obj_detail_cnts[3+i*table_elem_num], total_obj_detail_cnts[11+i*table_elem_num], total_obj_detail_cnts[19+i*table_elem_num], total_obj_detail_cnts[27+i*table_elem_num], total_obj_detail_cnts[35+i*table_elem_num]],
                                    ["left", total_obj_detail_cnts[4+i*table_elem_num], total_obj_detail_cnts[12+i*table_elem_num], total_obj_detail_cnts[20+i*table_elem_num], total_obj_detail_cnts[28+i*table_elem_num], total_obj_detail_cnts[36+i*table_elem_num]],
                                    ["rear_right", total_obj_detail_cnts[5+i*table_elem_num], total_obj_detail_cnts[13+i*table_elem_num], total_obj_detail_cnts[21+i*table_elem_num], total_obj_detail_cnts[29+i*table_elem_num], total_obj_detail_cnts[37+i*table_elem_num]],
                                    ["rear_left", total_obj_detail_cnts[6+i*table_elem_num], total_obj_detail_cnts[14+i*table_elem_num], total_obj_detail_cnts[22+i*table_elem_num], total_obj_detail_cnts[30+i*table_elem_num], total_obj_detail_cnts[38+i*table_elem_num]],
                                    ["rear", total_obj_detail_cnts[7+i*table_elem_num], total_obj_detail_cnts[15+i*table_elem_num], total_obj_detail_cnts[23+i*table_elem_num], total_obj_detail_cnts[31+i*table_elem_num], total_obj_detail_cnts[39+i*table_elem_num]]])
        print (tabulate(table[i], headers, tablefmt="pretty", numalign="right", colalign=("center", "center", "center", "center", "center")))


                        


if __name__ == "__main__":
    main()
