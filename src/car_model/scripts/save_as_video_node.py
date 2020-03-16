#!/usr/bin/env python
"""
Save images in a given topic as a video file.

Empirical statistics:
For images of width 608, height 384, encode them into a video of frame rate 20
for a fixed rosbag publish rate 20 and duration 30s:
- MJPG: 56.6MB
- PIM1: 29.3MB (MPEG-1) (frame drop, the final video is 25s)
- MP42: 34.5MB (MPEG-4)
"""
from __future__ import print_function
import argparse
import hashlib
import rospy
from cv_bridge import CvBridge
#other msg types : CompressedImage, PointCloud2
from sensor_msgs.msg import Image
import cv2


def _gen_node_name(topic_name):
    md5sum = hashlib.md5(topic_name).hexdigest()
    return "video_saver_node_" + md5sum[0:4]


class SaveAsVideoNode(object):
    def __init__(self, topic_name, imshow, frame_width, frame_height, fps, output):
        """
        topic_name -- Record images in the topic |topic_name|.
        imshow - Call cv2.imshow when receiving an image.
        frame_width - The width of an image.
        frame_height - The height of an image
        fps - The frame rate in the output video
        output - The output file name.
        """
        self.topic_name = topic_name
        self.node_name = _gen_node_name(topic_name)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.imshow = imshow
        self.cv_bridge = CvBridge()
        self.num_images = 0
        self.output_filename = output
        self.vdo = cv2.VideoWriter(self.output_filename,
                                   # cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   # cv2.VideoWriter_fourcc('P', 'I', 'M', '1'),
                                   cv2.VideoWriter_fourcc('M', 'P', '4', '2'),
                                   self.fps,
                                   (self.frame_width, self.frame_height))

        print("Record image topic {} in node {}".format(self.topic_name, self.node_name))
        rospy.init_node(self.node_name)
        rospy.Subscriber(self.topic_name, Image, self.img_cb)

    def img_cb(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        self.num_images += 1
        width = img.shape[1]
        height = img.shape[0]
        if self.frame_width != width or self.frame_height != height:
            print("Expect image size {}x{}, Got {}x{}".format(
                self.frame_width, self.frame_height, width, height))
        if self.imshow:
            cv2.imshow("image", img)
            cv2.waitKey(1)
        self.vdo.write(img)

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            rate.sleep()
        print("Shut down node {}".format(self.node_name))

        print("Write {}".format(self.output_filename))
        self.vdo.release()

        if self.imshow:
            cv2.destroyAllWindows()


def main():
    """Prog entry"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", "-t", required=True)
    parser.add_argument("--imshow", action="store_true")
    parser.add_argument("--frame-width", type=int, default=608)
    parser.add_argument("--frame-height", type=int, default=384)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", "-o", default="out.avi")
    args = parser.parse_args()

    node = SaveAsVideoNode(args.topic, args.imshow,
                           args.frame_width, args.frame_height,
                           args.fps, args.output)
    node.run()

if __name__ == '__main__':
    main()
