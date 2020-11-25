#!/usr/bin/env python3
import os
import glob
import rospy
from msgs.msg import DetectedObjectArray
from msgs.msg import DetectedObject
#from msgs.msg import PedObjectArray
#from msgs.msg import PedObject
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
from msgs.msg import DetectedObjectArray
from collections import deque
import threading
import time
import signal
import sys


import numpy as np
from data import DataSet
from extractor import Extractor
from keras.models import load_model



mutex = threading.Lock()

global_out_path = "./out"

front_bottom_60_cam_q = deque(maxlen=30)
front_bottom_60_obj_q = deque(maxlen=30)
front_top_far_30_cam_q = deque(maxlen=30)
front_top_far_30_obj_q = deque(maxlen=30)


seq_cnt = 20

q_mutex = threading.Lock()

#saved_LSTM_model = load_model("src/sensing/itri_drivenet/drivenet/data/cnn-lstm/lstm-features.061-0.000.hdf5")


def signal_handler(sig, frame):
	print('You pressed Ctrl+C!')
	global t
	t.do_run = False
	sys.exit(0)




def normalize(value, ori_min, ori_max, new_min, new_max):
	return (float(value) / (float(ori_max)-float(ori_min))) * (float(new_max)-float(new_min))

def find_nearest_obj(sec, nsec, obj_q):
	nearest = 1000

	for track2d_objs in reversed(obj_q):
		target_time = float(sec + (nsec/1000000000))
		obj_time = float(track2d_objs.header.stamp.secs + (track2d_objs.header.stamp.nsecs/1000000000))

		diff = abs(target_time - obj_time)
		if(diff < nearest):
			nearest = diff
			nearest_obj = track2d_objs

	return nearest_obj

def get_frame_cnt_recursive(path):
	if not os.path.exists(path):
		return 0
	frame_cnt = 0
	for root, dirs, files in os.walk(path):
		#print("root: %s" % root)
		#print("dirs: %s" % dirs)
		#print("files: %s" % files)
		for file in files:
			if ".jpg" in file:
				frame_cnt += 1

	return frame_cnt

def get_last_seq_idx(path):
	indexes = []

	if not os.path.exists(path):
		return -1

	dirs = os.listdir(path)

	#print(dirs)

	for dir in dirs:
		seq_idx = int(dir)
		if seq_idx not in indexes:
			indexes.append(seq_idx)

	if not indexes:
		return -1
	else:
		return(max(indexes))

def color_threshold(image):
	hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)	# Hue is range from 0 to 179

	#print(hls)

	mask1 = cv.inRange(hls, (0, 200, 25), (35, 255,255))
	mask2 = cv.inRange(hls, (150, 200, 25), (179, 255,255))

	mask = mask1 | mask2

	#cv.imwrite('mask.jpg', mask)

	## slice the output
	imask = mask>0
	output = np.zeros_like(image, np.uint8)
	output[imask] = image[imask]

	return output

def cam_cb(data, topic):

	q_mutex.acquire()

	if 'front_bottom_60' in topic:
		cam_q = front_bottom_60_cam_q
		obj_q = front_bottom_60_obj_q
	elif 'front_top_far_30' in topic:
		cam_q = front_top_far_30_cam_q
		obj_q = front_top_far_30_obj_q
	else:
		q_mutex.release()
		return

	cam_q.append(data)

	if(len(cam_q) < 20):
		q_mutex.release()
		return

	if(len(obj_q)==0):
		q_mutex.release()
		return

	cam_data = cam_q.popleft()

	#track2d_objs = obj_q.popleft()
	#print("cam_fb60: (%d, %d)" % (cam_data.header.stamp.secs, cam_data.header.stamp.nsecs))

	track2d_objs = find_nearest_obj(cam_data.header.stamp.secs, cam_data.header.stamp.nsecs, obj_q)

	#print("    find nearest obj: (%d, %d)" % (track2d_objs.header.stamp.secs, track2d_objs.header.stamp.nsecs))

	bridge = CvBridge()
	img = bridge.imgmsg_to_cv2(cam_data, desired_encoding="passthrough")

	#print("# of obj: %d" % len(track2d_objs.objects))

	for obj in track2d_objs.objects:
		if(obj.classId == 4): # Car
			cam_w = cam_data.width
			cam_h = cam_data.height
			track_x = obj.camInfo.u
			track_y = obj.camInfo.v
			track_w = obj.camInfo.width
			track_h = obj.camInfo.height
			#print("track obj: (%d, %d, %d, %d)" % (track_x, track_y, track_w, track_h))

			crop_x = int(normalize(track_x, 0, 1920, 0, cam_w))
			crop_y = int(normalize(track_y, 0, 1208, 0, cam_h))
			crop_w = int(float(track_w) / 1920.0 * float(cam_w))
			crop_h = int(float(track_h) / 1208.0 * float(cam_h))
			crop_img = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

			if crop_w < 30 or crop_h < 30:
				#print("too small picture..")
				continue	# skip "too small objects"


			global global_out_path

			out_path = global_out_path + "/" + os.path.basename(topic)

			out_path += "/ID%08d" % obj.track.id

			frame_cnt = get_frame_cnt_recursive(out_path)

			seq_idx = get_last_seq_idx(out_path)

			global seq_cnt
			if frame_cnt % seq_cnt == 0:
				seq_idx += 1

			#print("          frame_cnt: %d, seq_cnt: %d, seq_idx: %d" % (frame_cnt, seq_cnt, seq_idx))

			out_path += "/%08d" % (seq_idx)
			out_path2 = out_path + "_aaa"

			if not os.path.exists(out_path):
				os.makedirs(out_path)

			filename = "%d_%09d.jpg" % (cam_data.header.stamp.secs, cam_data.header.stamp.nsecs)
			file = out_path + "/" + filename

			print(file)

			#cv.imwrite(file, crop_img)

			final_img = color_threshold(crop_img)

			cv.imwrite(file, final_img)


	q_mutex.release()



def tracking2d_cb(data, topic):
	q_mutex.acquire()

	if 'front_bottom_60' in topic:
		cam_q = front_bottom_60_cam_q
		obj_q = front_bottom_60_obj_q
	elif 'front_top_far_30' in topic:
		cam_q = front_top_far_30_cam_q
		obj_q = front_top_far_30_obj_q
	else:
		q_mutex.release()
		return

	#now = time.time()
	obj_q.append(data)
	#print("# of obj: %d" % len(data.objects))
	#print("------- tracking2d: (%d, %d)" % (data.header.stamp.secs, data.header.stamp.nsecs))
	q_mutex.release()


def main():
	global global_out_path
	if (len(sys.argv) == 2):
		#print(sys.argv[1])
		global_out_path = sys.argv[1]

	print("create output dir: %s" % global_out_path)

	used_cam_topics = ['/cam/front_bottom_60', '/cam/front_top_far_30']
	used_track2d_topics = ['/Tracking2D/front_bottom_60', '/Tracking2D/front_top_far_30']

	rospy.init_node('bag2seq', anonymous=True)

	for topic in used_cam_topics:
		rospy.Subscriber(topic, Image, cam_cb, callback_args = topic)

	for topic in used_track2d_topics:
		rospy.Subscriber(topic, DetectedObjectArray, tracking2d_cb, callback_args = topic)

	#rospy.Subscriber("/Tracking2D/front_bottom_60", DetectedObjectArray, tracking2d_fb60_cb, callback_args = topic)
	#rospy.Subscriber("/cam_obj/front_bottom_60", DetectedObjectArray, tracking2d_fb60_cb, callback_args = my_topic)


	#print(normalize(960, 0, 1920, 0, 608))


	rospy.spin()

if __name__ == '__main__':
	main()


