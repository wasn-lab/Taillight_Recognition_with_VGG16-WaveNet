from __future__ import print_function
import argparse
import rosbag
import numpy as np
import cv2


def _save_avi(bag_filename, topic, avi_filename):
    bag = rosbag.Bag(bag_filename)
    avi = None
    #cv2.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
    for _topic, msg, timestamp in bag.read_messages(topics=[topic]):
        np_arr = np.fromstring(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if avi is None:
            width = image.shape[1]
            height = image.shape[0]
            fps = 15.0 
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            avi = cv2.VideoWriter(avi_filename, fourcc, fps, (width, height))
        avi.write(image)
    bag.close()
    if avi:
        avi.release()
    print("Write {}".format(avi_filename))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag", required=True)
    parser.add_argument("--topic", defualt="/xwin_grabber/rviz/jpg")
    parser.add_argument("--output", "-o", default="output.avi")
    args = parser.parse_args()
    _save_avi(args.rosbag, args.topic, args.output)

if __name__ == "__main__":
    main()
