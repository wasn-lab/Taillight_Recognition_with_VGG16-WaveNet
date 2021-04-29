#!/usr/bin/env python
"""
Calculate the publishing hz of given topics.
"""
from __future__ import print_function
import argparse
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class ImagePublishNode(object):
    def __init__(self, image_file, topic, fps):
        self.image = cv2.imread(image_file)
        self.topic = topic
        self.fps = fps
        rospy.init_node("ImagePublishNode")
        rospy.logwarn("Publish %s %s at %s, fps=%d", image_file, self.image.shape, topic, fps)

    def run(self):
        rate = rospy.Rate(self.fps)
        publisher = rospy.Publisher(self.topic, Image, queue_size=1)
        msg = CvBridge().cv2_to_imgmsg(self.image, encoding="bgr8")
        while not rospy.is_shutdown():
            publisher.publish(msg)
            rate.sleep()


def main():
    """Prog entry"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-file", "-f", required=True, help="Input image")
    parser.add_argument("--topic", "-t", required=True, help="Topic name")
    parser.add_argument("--fps", type=int, default=20, help="FPS")
    args, __unknown = parser.parse_known_args()

    node = ImagePublishNode(args.image_file, args.topic, args.fps)
    node.run()

if __name__ == '__main__':
    main()
