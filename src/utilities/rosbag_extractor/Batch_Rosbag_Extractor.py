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
# from msgs.msg import Rad
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

# ===== Control Variable ====
first_or_last_msg_in_a_sec = "first"  # first or last

bridge = [CvBridge(), CvBridge(), CvBridge(),
          CvBridge(), CvBridge(), CvBridge(),
          CvBridge(), CvBridge(), CvBridge()]

check_first = True

cam_topics = [
    '/cam/front_bottom_60',
    '/cam/front_top_far_30',
    '/cam/front_top_close_120',
    '/cam/right_front_60',
    '/cam/right_back_60',
    '/cam/left_front_60',
    '/cam/left_back_60',
    '/cam/back_top_120']


def read_topic_return_msg(msg, type, contain):
    if type == 0:  # lidar
        pc = pypcd.PointCloud.from_msg(msg)  # pylint: disable=no-member
        return pc

    if type == 1:  # camera
        try:
            img = contain.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            print(e)
        return img

    if type == 2:  # radar
        rad_points = msg.radPoint
        rad_points_list = []
        for one_point in rad_points:
            tempStrList = str(one_point).split("\n")
            tempStr = tempStrList[0] + ", " + tempStrList[1] + \
                ", " + tempStrList[2] + ", " + tempStrList[3] + "\n"
            rad_points_list.append(tempStr)
        return rad_points_list

    if type == 3:  # tf
        tf_transforms = msg.transforms
        tf_transforms_list = []
        tf_transforms_list.append(
            'stamp_secs, stamp_nsecs, source_frame_id, target_frame_id, translation_x, translation_y, translation_z, rotation_x, rotation_y, rotation_z, rotation_w\n')
        has_tf_from_baselink_to_map = False
        for one_transform in tf_transforms:
            if one_transform.child_frame_id != '/base_link' or one_transform.header.frame_id != '/map':
                continue
            tempStr = str(one_transform.header.stamp.secs) + ", " + str(one_transform.header.stamp.nsecs) + ", " + one_transform.child_frame_id + ", " + one_transform.header.frame_id +\
                ", " + str(one_transform.transform.translation.x) + ", " + str(one_transform.transform.translation.y) + \
                ", " + str(one_transform.transform.translation.z) + ", " + str(one_transform.transform.rotation.x) + \
                ", " + str(one_transform.transform.rotation.y) + ", " + str(one_transform.transform.rotation.z) + \
                ", " + str(one_transform.transform.rotation.w) + "\n"
            tf_transforms_list.append(tempStr)
            has_tf_from_baselink_to_map = True
        if has_tf_from_baselink_to_map:
            return tf_transforms_list

    if type == 4:  # veh_info
        veh_info_list = []
        veh_info_list.append(
            'stamp_secs, stamp_nsecs, ego_x, ego_y, ego_z, ego_heading, ego_speed, yaw_rate\n')
        tempStr = str(msg.header.stamp.secs) + ", " + str(msg.header.stamp.nsecs) + ", " + str(msg.ego_x) + ", " + str(msg.ego_y) +\
            ", " + str(msg.ego_z) + ", " + str(msg.ego_heading) + \
            ", " + str(msg.ego_speed) + ", " + str(msg.yaw_rate) + "\n"
        veh_info_list.append(tempStr)
        return veh_info_list


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

    if type == 1:  # camera
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

    if type == 2:  # radar
        if not os.path.exists(outdir + pathname + "_radar"):
            os.makedirs(outdir + pathname + "_radar")
        with open("{}{}{}.txt".format(outdir, pathname + "_radar/", filename), "w") as f:
            for line in msg:
                f.write(line)
        print("{}.txt ok.".format(filename))

    if type == 3:  # tf
        if msg:
            if not os.path.exists(outdir + pathname + "_tf"):
                os.makedirs(outdir + pathname + "_tf")
            with open("{}{}{}.csv".format(outdir, pathname + "_tf/", filename), "w") as f:
                for line in msg:
                    f.write(line)
            print("{}.csv ok.".format(filename))

    if type == 4:  # veh_info
        if msg:
            if not os.path.exists(outdir + pathname + "_veh_info"):
                os.makedirs(outdir + pathname + "_veh_info")
            with open("{}{}{}.csv".format(outdir, pathname + "_veh_info/", filename), "w") as f:
                for line in msg:
                    f.write(line)
            print("{}.csv ok.".format(filename))


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
        str_in = txt_input('Input FPS (1, 2, 5, 10):\n')
    except EOFError:
        str_in = "1"
    try:
        fps_input = int(str_in)
        if fps_input != 1 and fps_input != 2 and fps_input != 5 and fps_input != 10:
            fps_input = 1

        fps_inv = 10
        if fps_input == 1:
            fps_inv = 10
        elif fps_input == 2:
            fps_inv = 5
        elif fps_input == 5:
            fps_inv = 2
        elif fps_input == 10:
            fps_inv = 1

        print("FPS %d selected.\n", fps_input)

    except BaseException:
        fps_input = None
        exit()

    print ("** Start Extracting**")
    print("** Now is '{}_msg_in_sec' Mode!".format(first_or_last_msg_in_a_sec))
    rootDir = dir_path + '/bags'
    for dirName, subdirList, fileList in os.walk(rootDir):
        # print('Found directory: %s' % dirName)
        # os.chdir(dirName)
        for fname in fileList:
            # print('\t%s' % fname)
            if fname.endswith(".bag"):
                if dirName in Extracted_Dir_list:
                    pass
                else:
                    Extracted_Dir_list.append(dirName)
                    os.chdir(dirName)

                    # dir_lst = sorted(os.listdir(os.getcwd()))
                    # print (dir_lst)
                    dir_cur = os.getcwd()
                    print ("Current Directory: %s" % dir_cur)
                    # for dir_name in dir_lst:
                    print ("File in Directory: %s" % dir_cur)
                    if os.path.isdir(dir_cur) and not(
                            dirName.startswith(".")):
                        # print(dir_path + "/" + dir_name)
                        # lst = sorted(glob.glob(dir_path + "/bags/" + dir_name), key=getsize)
                        lst = sorted(os.listdir(os.getcwd()))
                        # print (lst)
                        for bagFile in lst:
                            os.chdir(dirName)
                            if bagFile.endswith(".bag"):
                                print (bagFile)
                                filename = bagFile.split(
                                    "/")[-1].split(".")[0]
                                print (filename)
                                # path = bagFile.split(filename)[0]

                                bag = rosbag.Bag((bagFile))
                                # topics_lst = bag.get_type_and_topic_info().topics

                                # check save path
                                outdir = dir_path + "/Extracted/" + filename + '/'
                                if not os.path.exists(outdir):
                                    os.makedirs(outdir)

                                # main loop
                                for topic, msg, t in bag.read_messages():
                                    if topic == "/LidarAll":
                                        if fps_input == 1:
                                            timestr = str(
                                                msg.header.stamp.secs)
                                        else:
                                            timestr = str(msg.header.stamp.to_nsec(
                                            ) / (fps_inv * 100000000) * fps_inv)

                                        lidarall_msg = read_topic_return_msg(
                                            msg, 0, "")
                                        if check_first:
                                            datestr_lidar = filename + "_lidar/"
                                            if os.path.exists(
                                                    outdir + datestr_lidar + timestr + '.pcd'):
                                                pass
                                            else:
                                                try:
                                                    read_msg_and_save(
                                                        lidarall_msg, 0, outdir, timestr, filename)
                                                except BaseException:
                                                    print(
                                                        "LidarAll cant save.")

                                    if topic in cam_topics:
                                        if fps_input == 1:
                                            timestr = str(
                                                msg.header.stamp.secs)
                                        else:
                                            timestr = str(msg.header.stamp.to_nsec(
                                            ) / (fps_inv * 100000000) * fps_inv)

                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 0, cam_topics[0], "a0", "A0")
                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 1, cam_topics[1], "a1", "A1")
                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 3, cam_topics[2], "b0", "B0")
                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 4, cam_topics[3], "b1", "B1")
                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 5, cam_topics[4], "b2", "B2")
                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 6, cam_topics[5], "c0", "C0")
                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 7, cam_topics[6], "c1", "C1")
                                        read_topic_and_save_camera(
                                            topic, msg, outdir, filename, timestr, 8, cam_topics[7], "c2", "C2")

                                    if topic == "/RadFront":
                                        if fps_input == 1:
                                            timestr = str(
                                                msg.radHeader.stamp.secs)
                                        else:
                                            timestr = str(msg.radHeader.stamp.to_nsec(
                                            ) / (fps_inv * 100000000) * fps_inv)

                                        rad_msg = read_topic_return_msg(
                                            msg, 2, "")
                                        if check_first:
                                            datestr_radar = filename + "_radar/"
                                            if os.path.exists(
                                                    outdir + datestr_radar + timestr + '_rad.txt'):
                                                pass
                                            elif timestr == "":
                                                pass
                                            else:
                                                try:
                                                    read_msg_and_save(
                                                        rad_msg, 2, outdir, timestr + "_rad", filename)
                                                except BaseException:
                                                    print("rad cant save.")

                                    if topic == "/tf":
                                        if fps_input == 1:
                                            timestr = str(
                                                msg.transforms[0].header.stamp.secs)
                                        else:
                                            timestr = str(msg.transforms[0].header.stamp.to_nsec(
                                            ) / (fps_inv * 100000000) * fps_inv)

                                        tf_msg = read_topic_return_msg(
                                            msg, 3, "")
                                        if check_first:
                                            datestr_tf = filename + "_tf/"
                                            if os.path.exists(
                                                    outdir + datestr_tf + timestr + '_tf.csv'):
                                                pass
                                            elif timestr == "":
                                                pass
                                            else:
                                                try:
                                                    read_msg_and_save(
                                                        tf_msg, 3, outdir, timestr + "_tf", filename)
                                                except BaseException:
                                                    print("tf cant save.")

                                    if topic == "/veh_info":
                                        if fps_input == 1:
                                            timestr = str(
                                                msg.header.stamp.secs)
                                        else:
                                            timestr = str(msg.header.stamp.to_nsec(
                                            ) / (fps_inv * 100000000) * fps_inv)

                                        veh_info_msg = read_topic_return_msg(
                                            msg, 4, "")
                                        if check_first:
                                            datestr_veh_info = filename + "_veh_info/"
                                            if os.path.exists(
                                                    outdir + datestr_veh_info + timestr + '_veh_info.csv'):
                                                pass
                                            elif timestr == "":
                                                pass
                                            else:
                                                try:
                                                    read_msg_and_save(
                                                        veh_info_msg, 4, outdir, timestr + "_veh_info", filename)
                                                except BaseException:
                                                    print(
                                                        "veh_info cant save.")
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
