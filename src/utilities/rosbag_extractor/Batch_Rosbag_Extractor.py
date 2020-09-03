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
import pypcd  # pylint: disable=import-error
from pypcd import PointCloud  # pylint: disable=import-error, no-name-in-module
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
from os.path import getsize
#from msgs.msg import Rad
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

# ===== Control Variable ====
first_or_last_msg_in_a_sec = "first"  # first or last

bridge = [CvBridge(), CvBridge(), CvBridge(),
          CvBridge(), CvBridge(), CvBridge(),
          CvBridge(), CvBridge(), CvBridge()]

check_first = True

cam_topics = ['/cam/front_bottom_60', '/cam/front_top_far_30',
    '/cam/front_top_close_120', '/cam/right_front_60', '/cam/right_back_60',
    '/cam/left_front_60', '/cam/left_back_60', '/cam/back_top_120']


def read_topic_return_msg(msg, type, contain):
    if type == 0:  # lidar
        pc = pypcd.PointCloud.from_msg(msg)  # pylint: disable=no-member
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


def read_msg_and_save(msg, type, outdir, filename, pathname=None, campos=None):
    if type == 0:  # lidar
        if not os.path.exists(outdir + pathname + "_lidar"):
            os.makedirs(outdir + pathname + "_lidar")
        msg.save_pcd(
            outdir +
            pathname +
            "_lidar/" +
            filename +
            '.pcd',
            compression='ascii')
        print("{}.pcd ok.".format(filename))

    if type == 1:  # cam
        if not os.path.exists(outdir + pathname + "_camera_" + campos):
            os.makedirs(outdir + pathname + "_camera_" + campos)
        cv2.imwrite(
            outdir +
            pathname +
            "_camera_" +
            campos +
            "/" +
            filename +
            '_camera_' +
            campos +
            '.jpg',
            msg)
        print("{}.jpg ok.".format(filename + '_camera_' + campos))

    if type == 2:  # rad
        if not os.path.exists(outdir + pathname + "_radar"):
            os.makedirs(outdir + pathname + "_radar")
        with open("{}{}{}.txt".format(outdir, pathname+"_radar/" ,filename), "w") as f:
            for line in msg:
                f.write(line)
        print("{}.txt ok.".format(filename))


def read_topic_and_save_camera(
        in_topicname,
        in_msg,
        in_outdir,
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
                in_outdir +
                datestr_cam_a0 +
                in_timestr +
                "_camera_" +
                in_append1 +
                    ".jpg"):
                pass
            elif in_timestr == "":
                pass
            else:
                try:
                    read_msg_and_save(
                        cam_msg,
                        1,
                        in_outdir,
                        in_timestr,
                        in_filename,
                        in_append1)
                except BaseException:
                    print
                    print(
                        "Cam" +
                        in_append2 +
                        " (" +
                        in_topicname +
                        ") cant save.")


# ===== main =====
def main():
    # ===== Global Variable =====
    dir_path = os.path.dirname(os.path.abspath('__file__'))
    os.chdir(os.path.dirname(os.path.abspath('__file__')) + "/bags")

    last_timestr = ""
    Extracted_Dir_list = ["Dummy"]

    timestr = ""
    # datestr = ""
    time_tick = False

    # User selection
    try:
        txt_input = raw_input
    except NameError:
        txt_input = input
    str_in = ""
    try:
        str_in = txt_input('Mode(1: in 1FPS, 2: in 10FPS):\n')
    except EOFError:
        str_in = "1"
    try:
        id_in = int(str_in)
        if id_in == 1:
            print("Mode 1 selected: 1FPS.\n")
        else:
            print("Mode 2 selected: 10FPS.\n")
    except BaseException:
        id_in = None
        exit()

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

                    #dir_lst = sorted(os.listdir(os.getcwd()))
                    #print (dir_lst)
                    dir_cur = os.getcwd()
                    print ("Current Directory: %s" % dir_cur)
                    #for dir_name in dir_lst:
                    print ("File in Directory: %s" % dir_cur)
                    if os.path.isdir(dir_cur) and not(
                            dirName.startswith(".")):
                        #print(dir_path + "/" + dir_name)
                        #lst = sorted(glob.glob(dir_path + "/bags/" + dir_name), key=getsize)
                        lst = sorted(os.listdir(os.getcwd()))
                        #print (lst)
                        for bagFile in lst:
                            os.chdir(dirName)
                            if bagFile.endswith(".bag"):
                                print (bagFile)
                                filename = bagFile.split(
                                    "/")[-1].split(".")[0]
                                print (filename)
                                #path = bagFile.split(filename)[0]

                                bag = rosbag.Bag((bagFile))
                                #topics_lst = bag.get_type_and_topic_info().topics

                                # check save path
                                outdir = dir_path + "/Extracted/" + filename + '/'
                                if not os.path.exists(outdir):
                                    os.makedirs(outdir)

                                # wtf main loop
                                for topic, msg, t in bag.read_messages():
                                    # if topic == "/LidarAll":
                                    #     print("A: " + topic)
                                    # Update LidarAll time
                                    if topic == "/LidarAll":
                                        if id_in == 1:
                                            timestr = str(msg.header.stamp.secs)
                                        else:
                                            timestr = str(msg.header.stamp.to_nsec()/100000000)
                                        lidarall_msg = read_topic_return_msg(msg, 0, "")
                                        if check_first:
                                            datestr_lidar = filename + "_lidar/"
                                            if os.path.exists(outdir + datestr_lidar + timestr + '.pcd'):
                                                pass
                                            else:
                                                try:
                                                    read_msg_and_save(lidarall_msg, 0, outdir, timestr, filename)
                                                except BaseException:
                                                    print("LidarAll cant save.")
                                    if topic in cam_topics:
                                        # pc = PointCloud.from_msg(msg)
                                        # timeint = msg.header.stamp.secs
                                        # datearray = time.localtime(timeint)
                                        # datestr = time.strftime(
                                        #     "%Y-%m-%d-%H-%M-%S", datearray)
                                        if id_in == 1:
                                            timestr = str(msg.header.stamp.secs)
                                        else:
                                            timestr = str(msg.header.stamp.to_nsec()/100000000)
                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 0, "/cam/front_bottom_60", "a0", "A0")
                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 1, "/cam/front_top_far_30", "a1", "A1")
                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 3, "/cam/front_top_close_120", "b0", "B0")
                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 4, "/cam/right_front_60", "b1", "B1")
                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 5, "/cam/right_back_60", "b2", "B2")
                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 6, "/cam/left_front_60", "c0", "C0")

                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 7, "/cam/left_back_60", "c1", "C1")
                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 8, "/cam/back_top_120", "c2", "C2")
                                    if topic == "/RadFront":
                                        if id_in == 1:
                                            timestr = str(msg.radHeader.stamp.secs)
                                        else:
                                            timestr = str(msg.radHeader.stamp.to_nsec()/100000000)
                                        rad_msg = read_topic_return_msg(msg, 2, "")
                                        if check_first:
                                            datestr_radar = filename + "_radar/"
                                            if os.path.exists(outdir + datestr_radar + timestr + '_rad.txt'):
                                                pass
                                            elif timestr == "":
                                                pass
                                            else:
                                                try:
                                                    read_msg_and_save(rad_msg, 2, outdir, timestr + "_rad", filename)
                                                except BaseException:
                                                    print("rad cant save.")
    print ("** Finish Extracting **")


if __name__ == "__main__":
    # ===== Global Functions =====
    if first_or_last_msg_in_a_sec.strip().lower() == "first":
        check_first = True
    elif first_or_last_msg_in_a_sec.strip().lower() == "last":
        check_first = False
    else:
        check_first = False
    main()
