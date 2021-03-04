# for validation of noise filter
import sys
import glob
import os
import time
import math
import string
import rospy
import pypcd
from pypcd import PointCloud
from sensor_msgs.msg import PointCloud2
#from msgs.msg import PointXYZIL

def pcd_to_lidarAll(input_path):
    pc = PointCloud.from_path(input_path)
    out_msg = pc.to_msg()
    out_msg.header = int(input_path.split("/")[-1].split(".")[0])
    return out_msg

def filtered_lidarAll_to_pcd(msg):
    timestr = str(msg.header.stamp.secs)

def main():
    rospy.init_node('labeled_pcd_filter')
    pub = rospy.Publisher('/LidarAll', PointCloud2, queue_size=10)
    sub = rospy.Subscriber('/LidarAll_filtered', PointCloud2, filtered_lidarAll_to_pcd, queue_size=10)
    traversed_dir_list = ["Dummy"]
    print ("**Start Filtering**")
    current_dir_path = os.path.dirname(os.path.abspath('__file__'))
    os.chdir(current_dir_path)
    root_dir = current_dir_path
    for dir_name, sub_dir_list, file_list in os.walk(root_dir):
        for fname in file_list:
            if fname.endswith(".pcd"):
                if dir_name in traversed_dir_list:
                    pass
                else:
                    traversed_dir_list.append(dir_name)
                    os.chdir(dir_name)

                    current_dir = os.getcwd()
                    print ("Current Directory: %s" % current_dir)
                    print ("File in Directory: %s" % current_dir)
                    if os.path.isdir(current_dir) and not (dir_name.startswith(".")):
                        pcd_list = sorted(os.listdir(os.getcwd()))
                        for pcd_file in pcd_list:
                            if pcd_file.endswith(".pcd"):
                                print ("Current File:%s" % pcd_file)
                                file_name = pcd_file.split("/")[-1].split(".")[0]
                                print (file_name)

                                pcd_path = current_dir + "/" + pcd_file
                                ori_msg = pcd_to_lidarAll(pcd_path)
                                pub.publish(ori_msg)
                                


if __name__ == "__main__":
    main()

