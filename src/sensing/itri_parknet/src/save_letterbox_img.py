"""
Get images of size 608x384, copy border in black to 608x608, and save them
in disk.
"""
from __future__ import print_function
import argparse
import logging
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

BRIDGE = CvBridge()
COLOR_BLACK = [0, 0, 0]


def _make_letterbox(img):
    rows, cols, _channels = img.shape
    if rows != 384 or cols != 608:
        logging.warning("img width/height is %d/%d, not 608/384. Skip it.",
                        cols, rows)
        return None
    top = (608 - rows) / 2
    bottom = top
    left = 0
    right = 0
    dst = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, None, COLOR_BLACK)
    rows, cols, _channels = dst.shape
    if rows != 608 or cols != 608:
        logging.error("dst width/height is %d/%d, not 608/608", cols, rows)
        return None
    return dst


def _image_callback(msg, prefix):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = BRIDGE.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, _e:
        print(_e)
    else:
        # Save your OpenCV2 image as a jpeg
        img = _make_letterbox(cv2_img)
        if img is None:
            return
        time = msg.header.stamp
        filename = prefix + str(time) + '.jpg'
        print("Write {}".format(filename))
        cv2.imwrite(filename, img)


def main():
    """Prog entry"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", "-t", default="/gmsl_camera/5")
    parser.add_argument("--prefix", "-p", default="front_")
    args = parser.parse_args()

    rospy.init_node('image_listener')
    image_topic = args.topic
    rospy.Subscriber(image_topic, Image, _image_callback, args.prefix)
    rospy.spin()


if __name__ == '__main__':
    main()
