#!/usr/bin/env python3
import sys
import rospy
extrenal_pkg_path = rospy.get_param("/vehicle_signal_detection/extrenal_pkg_path")
#sys.path.insert(1, '/home/chienminglo/workspace/itriadv/src/sensing/itri_vehicle_signal/scripts/install/lib/python3/dist-packages')
sys.path.insert(1, extrenal_pkg_path)


import os
import glob
from cv_bridge import CvBridge
import cv2 as cv
from collections import deque
import threading
import time
import shutil
import signal
from keras.models import load_model
from keras.preprocessing import image

import numpy as np
from extractor import Extractor

#from msgs.msg import PedObjectArray
#from msgs.msg import PedObject
from sensor_msgs.msg import Image
from msgs.msg import DetectedObjectArray
from msgs.msg import DetectedObject
from msgs.msg import VehicleObjectArray
from msgs.msg import VehicleObject
from msgs.msg import VehicleSignal



front_bottom_60_cam_q = deque(maxlen=30)
front_bottom_60_obj_q = deque(maxlen=30)
front_bottom_60_obj_q2 = deque(maxlen=10)
front_top_far_30_cam_q = deque(maxlen=30)
front_top_far_30_obj_q = deque(maxlen=30)

seq_cnt = 20

real_seq_cnt = 10

q_mutex = threading.Lock()

global_out_path = "/tmp/vehicle_signal_out"

last_frame_sec=0

g_pred_dict = {}

g_is_model_ready = False

def blend_icon(src, icon, location):
	# parameters
	x_start, y_start = location
	rows ,cols, channels = icon.shape

	# get roi
	roi = src[int(y_start):int(y_start) + rows, int(x_start):int(x_start) + cols]

	# get icon mask and mask inverse
	img2gray = cv.cvtColor(icon, cv.COLOR_BGR2GRAY)
	ret, mask = cv.threshold(img2gray, 254, 255, cv.THRESH_BINARY)
	mask_inv = cv.bitwise_not(mask)

	# dig black hole in src
	try:
		img1_bg = cv.bitwise_and(roi, roi, mask = mask)
	except:
		print('fail to blend icon at location: (%d, %d)' % location)
		return src

	# fill black in the icon border
	img2_fg = cv.bitwise_and(icon, icon, mask = mask_inv)

	# blend src and icon
	dst = cv.add(img1_bg, img2_fg)

	result = src.copy()

	result[int(y_start):int(y_start)+rows, int(x_start):int(x_start)+cols] = dst

	return result

def draw_icon(image, pred_str, crop_x, crop_y):
	right_icon_file = rospy.get_param("/vehicle_signal_detection/right_icon")
	warning_icon_file = rospy.get_param("/vehicle_signal_detection/warning_icon")

	right_icon = cv.imread(right_icon_file)
	warning_icon = cv.imread(warning_icon_file)

	if pred_str == 'turn_right':
		return blend_icon(image, right_icon, (crop_x, crop_y))
	elif pred_str == 'turn_left':
		left_icon = cv.flip(right_icon, 1)
		return blend_icon(image, left_icon, (crop_x, crop_y))
	elif pred_str == 'flashers':
		return blend_icon(image, warning_icon, (crop_x, crop_y))
	else:
		return image

def signal_handler(sig, frame):
	print('You pressed Ctrl+C!')
	global t
	t.do_run = False
	sys.exit(0)

def resize_image(image, w, h):
	resized_image = cv.resize(image, (w, h), interpolation = cv.INTER_AREA)
	return resized_image

def resize_frames(frames, w, h):
	resized_frames = []
	for frame in frames:
		# print("type:", end="");print(type(frame))
		# print(frame)
		resized_frame = cv.resize(frame, (w, h), interpolation = cv.INTER_AREA)
		resized_frames.append(resized_frame)

	return resized_frames

def save_cv2_img(file_path, image):
	dir = os.path.dirname(file_path)

	if not os.path.exists(dir):
		os.makedirs(dir)

	if not os.path.isfile(file_path):
		cv.imwrite(file_path, image)

def classify(saved_LSTM_model, extract_model, frames):

	resized_frames = resize_frames(frames, 224, 224)

	feature_sequence = []

	t1 = time.monotonic()

	# i = 0
	# for image in resized_frames:
		# if (1):  #if(i % 8 == 0):
			# feature = extract_model.extract_image(image)
		# i += 1
		# feature_sequence.append(feature)

		# #print("%d: " % i, end = " ");print(hash(feature[0]))

	feature_sequence = extract_model.extract_images(resized_frames)

	t2 = time.monotonic()

	#print('feature_sequence:', end=' ');print(len(feature_sequence))

	#print("       extract features cost time:", end=" "); print(t2-t1)

	prediction = saved_LSTM_model.predict(np.expand_dims(feature_sequence, axis=0))

	#print(prediction)

	t3 = time.monotonic()

	#print("       LSTM model cost time:", end=" "); print(t3-t2)

	pred = prediction[0]
	return pred

def prob2str(prob):
	if(max(prob) == prob[1]):
		return 'turn_right'
	elif(max(prob) == prob[2]):
		return 'turn_left'
	elif(max(prob) == prob[3]):
		return 'flashers'
	else:
		return 'no_signal'

def pub_obj_thread(arg):
	#print("pub_obj_thread~~~")

	pub = rospy.Publisher('/VehicleSignal/detect_object/front_bottom_60', VehicleObjectArray, queue_size=10)

	rate = rospy.Rate(40) # pick a large number, so the actual rate will almost the same as the pipeline source

	obj_q2 = front_bottom_60_obj_q2

	while getattr(t, "do_run", True) and (not rospy.is_shutdown()):
	#while not rospy.is_shutdown():
		q_mutex.acquire()

		if(len(obj_q2) > 0):
			track2d_objs = obj_q2.popleft()


			vehicle_objs = VehicleObjectArray()

			vehicle_objs.header = track2d_objs.header

			for _obj in track2d_objs.objects:
				vehicle_obj = VehicleObject()
				vehicle_obj.header = _obj.header
				vehicle_obj.classId = _obj.classId
				vehicle_obj.distance = _obj.distance
				vehicle_obj.absSpeed = _obj.absSpeed
				vehicle_obj.relSpeed = _obj.relSpeed
				vehicle_obj.bPoint = _obj.bPoint
				vehicle_obj.cPoint = _obj.cPoint
				vehicle_obj.bOrient = _obj.bOrient
				vehicle_obj.track = _obj.track
				vehicle_obj.fusionSourceId = _obj.fusionSourceId
				vehicle_obj.camInfo = _obj.camInfo
				vehicle_obj.radarInfo = _obj.radarInfo
				vehicle_obj.lidarInfo = _obj.lidarInfo

				# vehicleSignal
				if _obj.track.id in g_pred_dict.keys():
					pred, pred_str = g_pred_dict.get(_obj.track.id)
					if(pred_str == 'no_signal'):
						vehicle_obj.vehicleSignal.signal = 0
					elif(pred_str == 'turn_right'):
						vehicle_obj.vehicleSignal.signal = 1
					elif(pred_str == 'turn_left'):
						vehicle_obj.vehicleSignal.signal = 2
					elif(pred_str == 'flashers'):
						vehicle_obj.vehicleSignal.signal = 3
					else:
						vehicle_obj.vehicleSignal.signal = 4 # unknown
				else:
					vehicle_obj.vehicleSignal.signal = 4 # unknown

				vehicle_obj.vehicleSignal.brake = 2 # unknown
				vehicle_obj.vehicleSignal.intent = 5 # unknown
				vehicle_obj.vehicleSignal.view = 4 # unknown

				vehicle_objs.objects.append(vehicle_obj)

			pub.publish(vehicle_objs)


		q_mutex.release()
		rate.sleep()

def pub_img_thread(arg):
	#print("pub_img_thread~~~")

	pub = rospy.Publisher('/VehicleSignal/detect_image/front_bottom_60', Image, queue_size=10)

	rate = rospy.Rate(40) # pick a large number, so the actual rate will almost the same as the pipeline source

	obj_q = front_bottom_60_obj_q
	cam_q = front_bottom_60_cam_q

	while getattr(t, "do_run", True) and (not rospy.is_shutdown()):
	#while not rospy.is_shutdown():
		q_mutex.acquire()

		# print(len(cam_q))
		# print(len(obj_q))

		if(len(cam_q) > 0 and len(obj_q) > 0):
			#img_data = cam_q.popleft()
			img_data = cam_q[0]

			sec = img_data.header.stamp.secs
			nsec = img_data.header.stamp.nsecs
			track2d_objs = find_nearest_obj(sec, nsec, obj_q)

			bridge = CvBridge()
			cv_img = bridge.imgmsg_to_cv2(img_data, desired_encoding="passthrough")

			for obj in track2d_objs.objects:
				if(obj.classId == 4): # Car
					cam_w = img_data.width
					cam_h = img_data.height
					track_x = obj.camInfo.u
					track_y = obj.camInfo.v
					track_w = obj.camInfo.width
					track_h = obj.camInfo.height
					#print("track obj: (%d, %d, %d, %d)" % (track_x, track_y, track_w, track_h))

					crop_x = int(normalize(track_x, 0, 1920, 0, cam_w))
					crop_y = int(normalize(track_y, 0, 1208, 0, cam_h))
					crop_w = int(float(track_w) / 1920.0 * float(cam_w))
					crop_h = int(float(track_h) / 1208.0 * float(cam_h))

					#cv.rectangle(cv_img, (crop_x, crop_y), (crop_x+crop_w, crop_y+crop_h), (0, 255, 0), 2)

					try:
						pred, pred_str = g_pred_dict.get(obj.track.id)
					except:
						pred = None
						pred_str = None
						continue

					icon_x = crop_x + (crop_w/2) - 15
					icon_y = crop_y + (crop_h/2) - 15
					#print('    ', end=' ');print(pred, end=' ');print(max(pred))
					pred_prob = '%.2f' % max(pred)
					cv.putText(cv_img, pred_prob, (int(icon_x), int(icon_y)-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), thickness=1,lineType=cv.LINE_AA)
					cv_img = draw_icon(cv_img, pred_str, int(icon_x), int(icon_y))


					#print("    get pred: track id: %d, pred_str: %s" % (obj.track.id, pred_str))
					


							

			out_img = bridge.cv2_to_imgmsg(cv_img, encoding="passthrough")
			pub.publish(out_img)

		q_mutex.release()
		rate.sleep()

def remove_files(dir, files):
	for file in files:
		f_path = dir + '/' + file
		try:
			os.remove(f_path)
		except OSError:
			pass

def clean_old_sequences(save_time_sec):
	expired_time = int(time.time()) - save_time_sec

	print('remove files before: %d' % expired_time)

	for root, dirs, files in os.walk(global_out_path):
		# print(root)
		# print(dirs)
		# print(files)
		#os.remove(file)

		del_files = []
		del_files_cnt = 0

		for file in files:
			if(len(file) < seq_cnt):  # skip label files 'no_signal' 'turn_left' 'turn_right' 'flashers'
				continue
			file_no_ext = file.split('.')[0]
			timestamp = int(file_no_ext.split('_')[0])
			#print(timestamp)

			if(timestamp < expired_time):
				del_files += [file]
				del_files_cnt += 1

			if(del_files_cnt >= 1):
				del_files += ['turn_left', 'turn_right', 'no_signal', 'flashers']

			#print('will remove', end=' ');print(del_files)
			remove_files(root, del_files)

		if(len(os.listdir(root)) == 0):
			shutil.rmtree(root)

	for root,_,_ in os.walk(global_out_path):
		#print(root, end=' ');print(len(os.listdir(root)))
		if(len(os.listdir(root)) == 0):
			shutil.rmtree(root)


def model_thread(arg):
	#print("model_thread~~~")

	global t
	t = threading.currentThread()

	extract_model = Extractor(image_shape=(224, 224, 3))

	model_file = rospy.get_param("/vehicle_signal_detection/model_file")

	saved_LSTM_model = load_model(model_file)

	out_path = "/home/chienminglo/workspace/itriadv/out/"

	global g_is_model_ready
	g_is_model_ready = True

	while getattr(t, "do_run", True):

		t1 = time.monotonic()

		global global_out_path
		for root, dirs, files in os.walk(global_out_path):
			#print(len(files))
			#print (root, dirs, files)

			frames = []

			if(len(files) == seq_cnt):
				track_id_dir = root.split('/')[-2]
				track_id = int(track_id_dir.split('_')[1])
				seq_idx = int(os.path.basename(root))
				topic = root.split('/')[-3]
				# print("seq_idx:", end=' ');print(seq_idx)					# ex: 3
				# print("track_id_dir:", end=' ');print(track_id_dir)		# ex: ID_00480798
				# print("topic:", end=' ');print(topic)						# ex: front_bottom_60

				for file in files:
					file_no_ext = file.split('.')[0]
					timestamp = int(file_no_ext.split('_')[0])
					sec = int(file_no_ext.split('_')[1])
					nsec = int(file_no_ext.split('_')[2])
					# print("time: (%d, %d)" % (sec, nsec))

					image_path = root + "/" + file
					pil_img = image.load_img(image_path)
					img = image.img_to_array(pil_img)

					frames.append(img)

				tt1 = time.monotonic()
				pred = classify(saved_LSTM_model, extract_model, frames)
				tt2 = time.monotonic()
				#print("   CNN-LSTM cost time:", end=" "); print(tt2-tt1)
				pred_str = prob2str(pred)
				classifed_result = root + "/" + pred_str
				open(classifed_result, 'a').close()
				print("        [%d]: %.2f | %.2f | %.2f | %.2f" % (track_id, pred[0], pred[1], pred[2], pred[3]))
				g_pred_dict.update({track_id:(pred, pred_str)})
				#print(g_pred_dict)

				# remove the folder!!!
				#print("\x1b[31m", end="");print("will remove [%s]" % root);print("\x1b[0m", end="")
				shutil.rmtree(root)

		# # clean sequences older than 30 sec
		# if (int(time.time()) % 10 == 0):
			# clean_old_sequences(30)

		time.sleep(0.1)

		t2 = time.monotonic()
		#print("   elapsed time:", end=" "); print(t2-t1)

	print("stopping model_thread")


def normalize(value, ori_min, ori_max, new_min, new_max):
	return (float(value) / (float(ori_max)-float(ori_min))) * (float(new_max)-float(new_min))

def find_nearest_obj(sec, nsec, obj_q):
	nearest = 1000

	if(len(obj_q) == 0):
		return None

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

	global g_is_model_ready
	if(g_is_model_ready == False):
		q_mutex.release()
		return

	#cam_data = cam_q.popleft()
	cam_data = cam_q[0]


	#print("cam_fb60: (%d, %d)" % (cam_data.header.stamp.secs, cam_data.header.stamp.nsecs))

	track2d_objs = find_nearest_obj(cam_data.header.stamp.secs, cam_data.header.stamp.nsecs, obj_q)
	if(track2d_objs == None):
		return

	#print("    find nearest obj: (%d, %d)" % (track2d_objs.header.stamp.secs, track2d_objs.header.stamp.nsecs))

	bridge = CvBridge()
	img = bridge.imgmsg_to_cv2(cam_data, desired_encoding="passthrough")

	#print("# of obj: %d" % len(track2d_objs.objects))

	sec = cam_data.header.stamp.secs
	nsec = cam_data.header.stamp.nsecs

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
			#crop_img = 'crop img'

			if crop_w < 20 or crop_h < 20:
				#print("too small picture..")
				continue	# skip "too small objects"

			# save file
			global global_out_path
			out_path = global_out_path + "/" + os.path.basename(topic)
			out_path += "/ID_%08d" % obj.track.id
			frame_cnt = get_frame_cnt_recursive(out_path)
			seq_idx = get_last_seq_idx(out_path)
			global seq_cnt
			if frame_cnt % seq_cnt == 0:
				seq_idx += 1
			#print("          frame_cnt: %d, seq_cnt: %d, seq_idx: %d" % (frame_cnt, seq_cnt, seq_idx))
			out_path += "/%08d" % (seq_idx)
			if not os.path.exists(out_path):
				os.makedirs(out_path)
			timestamp = int(time.time())
			filename = "%d_%d_%09d.jpg" % (timestamp, sec, nsec)
			file = out_path + "/" + filename
			#print(file)
			cv.imwrite(file, crop_img)

	q_mutex.release()



def tracking2d_cb(data, topic):
	q_mutex.acquire()

	if 'front_bottom_60' in topic:
		cam_q = front_bottom_60_cam_q
		obj_q = front_bottom_60_obj_q
		obj_q2 = front_bottom_60_obj_q2
	elif 'front_top_far_30' in topic:
		cam_q = front_top_far_30_cam_q
		obj_q = front_top_far_30_obj_q
	else:
		q_mutex.release()
		return

	obj_q.append(data)
	obj_q2.append(data)

	q_mutex.release()


def main():
	print("vehicle signal detection is running ..")

	global global_out_path
	if os.path.exists(global_out_path):
		shutil.rmtree(global_out_path)
	os.makedirs(global_out_path)

	used_cam_topics = ['/cam/front_bottom_60', '/cam/front_top_far_30']
	used_track2d_topics = ['/Tracking2D/front_bottom_60', '/Tracking2D/front_top_far_30']

	rospy.init_node('vehicle_signal_detection', anonymous=True)

	# consumer thread
	signal.signal(signal.SIGINT, signal_handler)

	for topic in used_cam_topics:
		rospy.Subscriber(topic, Image, cam_cb, callback_args = topic)

	for topic in used_track2d_topics:
		rospy.Subscriber(topic, DetectedObjectArray, tracking2d_cb, callback_args = topic)

	# object publisher thread
	t_pub_obj = threading.Thread(target = pub_obj_thread, args=("task",))
	t_pub_obj.start()
	#t_pub_obj.join()

	# image publisher thread
	t_pub_img = threading.Thread(target = pub_img_thread, args=("task",))
	t_pub_img.start()
	#t_pub_img.join()

	# model thread
	t_model = threading.Thread(target = model_thread, args=("task",))
	t_model.start()
	#t_model.join()


	rospy.spin()

if __name__ == '__main__':
	main()


