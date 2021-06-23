from __future__ import print_function
import argparse
import datetime
import rosbag
import numpy as np
import cv2


def _save_avi(bag_filename, topic, avi_filename):
    bag = rosbag.Bag(bag_filename)
#cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    images = []
    timestamps = []
    for _topic, msg, timestamp in bag.read_messages(topics=[topic]):
        timestamps.append(timestamp)
        np_arr = np.fromstring(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        images.append(image)
    bag.close()

    if images:
        width = images[0].shape[1]
        height = images[0].shape[0]

        first_dt = datetime.datetime.fromtimestamp(timestamps[0])
        last_dt = datetime.datetime.fromtimestamp(timestamps[-1])
        delta = last_dt - first_dt
        duration_in_second = delta.seconds
        if duration_in_second <= 0:
            duration_in_second = 1

        fps = len(images) / float(duration_in_second)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        avi = cv2.VideoWriter(avi_filename, fourcc, fps, (width, height))
        for image in images:
            avi.write(image)
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
