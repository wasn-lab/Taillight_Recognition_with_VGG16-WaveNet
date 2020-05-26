# Reference:
# rosbag: http://wiki.ros.org/rosbag/Cookbook
# msg_to_image:
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
import rosbag
import sys
import cv2
import glob
import os
import time
import string
import rospy
import pypcd
from pypcd import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
from os.path import getsize
#from msgs.msg import Rad
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

# ===== Control Variable ====
first_or_last_msg_in_a_sec = "first"  # first or last


# ===== Global Variable =====
dir_path = os.path.dirname(os.path.abspath('__file__'))
os.chdir(os.path.dirname(os.path.abspath('__file__')) + "/bags")
count = 0
current_filename = " "
last_extracted_pcd = " "
last_extracted_img_a0 = " "
last_extracted_img_a1 = " "
last_extracted_img_a2 = " "
last_extracted_img_b0 = " "
last_extracted_img_b1 = " "
last_extracted_img_b2 = " "
last_extracted_img_c0 = " "
last_extracted_img_c1 = " "
last_extracted_img_c2 = " "
last_extracted_rad = " "

last_timestr = ""
last_datestr = ""

bridge = [CvBridge(), CvBridge(), CvBridge(),
          CvBridge(), CvBridge(), CvBridge(),
          CvBridge(), CvBridge(), CvBridge()]

bags_lst = []
bags_dir_lst = []
bags_dir = ""
Extracted_Dir_list = ["Dummy"]

timestr = ""
datestr = ""
time_tick = False
# ===== Global Functions =====
if first_or_last_msg_in_a_sec.strip().lower() == "first":
    check_first = True
elif first_or_last_msg_in_a_sec.strip().lower() == "last":
    check_first = False
else:
    check_first = False


def read_topic_return_msg(msg, type, contain):
    if type == 0:  # lidar
        pc = PointCloud.from_msg(msg)
        return pc
    # if type == 1: #cam
    #     try:
    #         img = contain.compressed_imgmsg_to_cv2(msg,'bgr8')
    #     except CvBridgeError as e:
    #         print(e)
    #     return img
    if type == 1:  # cam
        try:
            img = contain.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            print(e)
        return img

    if type == 2:  # rad
        rad_points = msg.radPoint
        rad_points_list = []
        for one_point in rad_points:
            tempStrList = str(one_point).split("\n")
            tempStr = tempStrList[0] + ", " + tempStrList[1] + \
                ", " + tempStrList[2] + ", " + tempStrList[3] + "\n"
            rad_points_list.append(tempStr)
        return rad_points_list


def read_msg_and_save(mssg, type, filename, pathname=None, campos=None):
    if type == 0:  # lidar
        if not os.path.exists(outdir_pcd + pathname + "_lidar"):
            os.makedirs(outdir_pcd + pathname + "_lidar")
        mssg.save_pcd(
            outdir_pcd +
            pathname +
            "_lidar/" +
            filename +
            '.pcd',
            compression='ascii')
        print("{}.pcd ok.".format(filename))

    if type == 1:  # cam
        if not os.path.exists(outdir_img + pathname + "_camera_" + campos):
            os.makedirs(outdir_img + pathname + "_camera_" + campos)
        cv2.imwrite(
            outdir_img +
            pathname +
            "_camera_" +
            campos +
            "/" +
            filename +
            '_camera_' +
            campos +
            '.jpg',
            mssg)
        print("{}.jpg ok.".format(filename + '_camera_' + campos))

    if type == 2:  # rad
        with open("{}{}.txt".format(outdir_rad, filename), "w") as f:
            for line in mssg:
                f.write(line)
        print("{}.txt ok.".format(filename))


def read_topic_and_save_camera(
        in_topicname,
        in_msg,
        in_filename,
        in_timestr,
        in_bridge_id=0,
        target_topicname="/cam/front_bottom_60",
        in_append1="a0",
        in_append2="A0"):
    if in_topicname == target_topicname:
        cam_msg = read_topic_return_msg(in_msg, 1, bridge[in_bridge_id])
        if check_first:
            datestr_cam_a0 = in_filename + "_camera_" + in_append1 + "/"
            if os.path.exists(
                outdir_img +
                datestr_cam_a0 +
                in_timestr +
                "_camera_" +
                in_append1 +
                    ".jpg"):
                pass
            else:
                try:
                    read_msg_and_save(
                        cam_msg, 1, in_timestr, in_filename, in_append1)
                except BaseException:
                    print(
                        "Cam" +
                        in_append2 +
                        " (" +
                        in_topicname +
                        ") cant save.")


# ===== main =====
print ("** Start Extracting**")
print("** Now is '{}_msg_in_sec' Mode!".format(first_or_last_msg_in_a_sec))
rootDir = dir_path + '/bags'
for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName)
    # os.chdir(dirName)
    for fname in fileList:
        #print('\t%s' % fname)
        if fname.endswith(".bag"):
            if dirName in Extracted_Dir_list:
                pass
            else:
                Extracted_Dir_list.append(dirName)
                os.chdir(dirName)

                dir_lst = sorted(os.listdir(os.getcwd()))
                #print (dir_lst)
                dir_cur = os.getcwd()
                print ("Current Directory: %s" % dir_cur)
                for dir_name in dir_lst:
                    print ("File in Directory: %s" % dir_name)
                    if os.path.isdir(dir_cur) and not(
                            dir_name.startswith(".")):
                        #print(dir_path + "/" + dir_name)
                        #lst = sorted(glob.glob(dir_path + "/bags/" + dir_name), key=getsize)
                        lst = sorted(os.listdir(os.getcwd()))
                        #print (lst)
                        for bagFile in lst:
                            os.chdir(dirName)
                            if bagFile.endswith(".bag"):
                                print (bagFile)
                                filename = bagFile.split("/")[-1].split(".")[0]
                                print (filename)
                                path = bagFile.split(filename)[0]

                                bag = rosbag.Bag((bagFile))
                                topics_lst = bag.get_type_and_topic_info().topics

                                # check save path
                                outdir_pcd = dir_path + "/Extracted/" + filename + '/'
                                outdir_img = dir_path + "/Extracted/" + filename + '/'
                                outdir_rad = dir_path + "/Extracted/" + filename + '/'
                                if not os.path.exists(outdir_pcd):
                                    os.makedirs(outdir_pcd)
                                if not os.path.exists(outdir_img):
                                    os.makedirs(outdir_img)
                                if not os.path.exists(outdir_rad):
                                    os.makedirs(outdir_rad)

                                # wtf main loop
                                for topic, msg, t in bag.read_messages():
                                    #print("A: " + topic)
                                    # Update LidarAll time
                                    if topic == "/LidarAll":
                                        pc = PointCloud.from_msg(msg)
                                        timeint = msg.header.stamp.secs
                                        datearray = time.localtime(timeint)
                                        datestr = time.strftime(
                                            "%Y-%m-%d-%H-%M-%S", datearray)
                                        timestr = str(msg.header.stamp.secs)
                                        if timestr != last_timestr:
                                            time_tick = True
                                        else:
                                            time_tick = False
                                        # print(time_tick)

                                    if not time_tick:
                                        if topic == "/LidarAll":
                                            lidarall_msg = read_topic_return_msg(
                                                msg, 0, "")
                                            if check_first:
                                                datestr_lidar = filename + "_lidar/"
                                                if os.path.exists(
                                                        outdir_pcd + datestr_lidar + timestr + '.pcd'):
                                                    pass
                                                else:
                                                    try:
                                                        read_msg_and_save(
                                                            lidarall_msg, 0, timestr, filename)
                                                    except BaseException:
                                                        print(
                                                            "LidarAll cant save.")

                                        read_topic_and_save_camera(
                                            topic, msg, filename, timestr, 0, "/cam/front_bottom_60", "a0", "A0")
                                        read_topic_and_save_camera(
                                            topic, msg, filename, timestr, 1, "/cam/front_top_far_30", "a1", "A1")
                                        read_topic_and_save_camera(
                                            topic, msg, filename, timestr, 3, "/cam/front_top_close_120", "b0", "B0")
                                        read_topic_and_save_camera(
                                            topic, msg, filename, timestr, 4, "/cam/right_front_60", "b1", "B1")
                                        read_topic_and_save_camera(
                                            topic, msg, filename, timestr, 5, "/cam/right_back_60", "b2", "B2")
                                        read_topic_and_save_camera(
                                            topic, msg, filename, timestr, 6, "/cam/left_front_60", "c0", "C0")
                                        read_topic_and_save_camera(
                                            topic, msg, filename, timestr, 7, "/cam/left_back_60", "c1", "C1")
                                        read_topic_and_save_camera(
                                            topic, msg, filename, timestr, 8, "/cam/back_top_120", "c2", "C2")

                                        if topic == "/RadFront":
                                            rad_msg = read_topic_return_msg(
                                                msg, 2, "")
                                            if check_first:
                                                if os.path.exists(
                                                        outdir_rad + timestr + '_rad.txt'):
                                                    pass
                                                else:
                                                    try:
                                                        read_msg_and_save(
                                                            rad_msg, 2, timestr + "_rad")
                                                    except BaseException:
                                                        print("rad cant save.")

                                    elif time_tick:
                                        if check_first:
                                            pass
                                        else:
                                            # save first and update new
                                            time_tick = False
                                            try:
                                                read_msg_and_save(
                                                    lidarall_msg, 0, last_timestr, filename)
                                            except BaseException:
                                                print("lidar cant save.")

                                            try:
                                                read_msg_and_save(
                                                    camA0_msg, 1, last_timestr, filename, "a0")
                                            except BaseException:
                                                print("CamA0 cant save.")

                                            try:
                                                read_msg_and_save(
                                                    camA1_msg, 1, last_timestr, filename, "a1")
                                            except BaseException:
                                                print("CamA1 cant save.")

                                            # try:
                                            #     read_msg_and_save(camA2_msg, 1, last_timestr, filename, "a2")
                                            # except:
                                            #     print("CamA2 cant save.")
                                            try:
                                                read_msg_and_save(
                                                    camB0_msg, 1, last_timestr, filename, "b0")
                                            except BaseException:
                                                print("CamB0 cant save.")
                                            try:
                                                read_msg_and_save(
                                                    camB1_msg, 1, last_timestr, filename, "b1")
                                            except BaseException:
                                                print("CamB1 cant save.")
                                            try:
                                                read_msg_and_save(
                                                    camB2_msg, 1, last_timestr, filename, "b2")
                                            except BaseException:
                                                print("CamB2 cant save.")
                                            try:
                                                read_msg_and_save(
                                                    camC0_msg, 1, last_timestr, filename, "c0")
                                            except BaseException:
                                                print("CamC0 cant save.")
                                            try:
                                                read_msg_and_save(
                                                    camC1_msg, 1, last_timestr, filename, "c1")
                                            except BaseException:
                                                print("CamC1 cant save.")
                                            try:
                                                read_msg_and_save(
                                                    camC2_msg, 1, last_timestr, filename, "c2")
                                            except BaseException:
                                                print("CamC2 cant save.")
                                            try:
                                                read_msg_and_save(
                                                    rad_msg, 2, last_timestr + "_rad")
                                            except BaseException:
                                                print("rad cant save.")
                                    last_timestr = timestr
                                    last_datestr = datestr

print ("** Finish Extracting **")
