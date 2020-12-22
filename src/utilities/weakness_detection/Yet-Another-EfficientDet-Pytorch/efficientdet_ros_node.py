# Author: Zylo117, modified by ICL-U
"""
Simple Inference Script of EfficientDet-Pytorch
"""
import argparse
import json
import io
import logging
import os
import time

import torch
#from torch.backends import cudnn
#from matplotlib import colors
import numpy as np
import cv2
from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge, CvBridgeError

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import (preprocess, invert_affine, postprocess,
                         STANDARD_COLORS, standard_to_bgr, get_index_label,
                         plot_one_box)
from efficientdet_itri import EfficientDet


class EfficientDetRosNode(object):
    def __init__(self, coef, topic):
        rospy.init_node("EfficientDetRosNode")
        print("subscribe {}".format(topic))
        rospy.Subscriber(topic, Image, self._cb)

        print("Init EfficientDet with coef {}".format(coef))
#        self.net = EfficientDet(coef)
        print("Done init EfficientDet with coef {}".format(coef))
        self.cv_bridge = CvBridge()
        self.img = None

    def _cb(self, msg):
        print(type(msg))
        self.img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        print(self.img.shape)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

def main():
    logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--coef", type=int, default=4)
    parser.add_argument("--topic", default="/cam/front_bottom_60")
    args = parser.parse_args()
    node = EfficientDetRosNode(args.coef, args.topic)
    node.run()


if __name__ == "__main__":
    print("Work in progress. This script is unable to run at this moment.")
#    main()
