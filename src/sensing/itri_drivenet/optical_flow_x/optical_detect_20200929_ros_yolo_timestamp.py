import rospy
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import threading
import argparse
import time
import random
import cv2,os

import numpy as np
from opticalflow_motion_ros_20200828 import ObjectDetectionsWithOpticalFlow as optical
import sys 
sys.path.append('/home/jason/gitlab/itriadv/devel/lib/python2.7/dist-packages')
#sys.path.append('/home/lipei/u-ros-system/itriadv/devel/lib/python2.7/dist-packages')

from msgs.msg import DetectedObjectArray
queue = []


img_len = 0
yolo_len = 0
yolo_timestamp_len = 0
img_timestamp = []
img_list = []
keep_yolo_time = []
yolo_bbox_time = []
bridge = CvBridge()
full_image_bbox_info = []
keep_data_object = []
rospy.init_node('listener', anonymous=True)

colors = None

def ros_subscriber_image():
    rospy.Subscriber('/cam/front_bottom_60', Image, img_callback)
    rospy.spin()

def img_callback(data): 
    global img_timestamp
    global img_list
    img_list.append([data,data.header.stamp]) #bridge.imgmsg_to_cv2(data, "bgr8")) 
    img_timestamp.append(data.header.stamp)
    global img_len
    img_len += 1

def ros_subscriber_yolo_timestamp():    
    rospy.Subscriber('/cam_obj/front_bottom_60/time_info', Header, yolo_timestamp_callback)
    rospy.spin()

def yolo_timestamp_callback(data):
    global yolo_timestamp_len
    yolo_timestamp_len += 1
    global keep_yolo_time
    keep_yolo_time.append(data.stamp)    

    
def ros_subscriber_yolo():
    rospy.Subscriber('/cam_obj/front_bottom_60', DetectedObjectArray, yolo_callback)
    rospy.spin()

def yolo_callback(data):
    global yolo_len
    yolo_len += 1
    global yolo_bbox_time
    yolo_bbox_time.append(data.header.stamp)    
    global keep_data_object    
    keep_data_object.append(data.objects)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or int(round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)        

        
def debug_img(debug_img_publish,tracking_bbox_publish):
    global colors
    global full_image_bbox_info
    raw_img_size = np.array([1920 ,1208 ,1920 ,1208])
    yolo_img_size = np.array([608, 384 ,608, 384])
    size_scale = np.true_divide(raw_img_size,yolo_img_size)      
    while not rospy.is_shutdown():
        if len(full_image_bbox_info)>0:
            info1 = full_image_bbox_info[0]
            if info1[2] == "YOLO":
                for res in info1[1]:
                    xyxy = res['box']
                    cls = res['category']
                    plot_one_box(xyxy, info1[0], label=str(int(cls)), color=[225, 0, 0])
                cv2.putText(info1[0], "{}".format(info1[2]), (10, 30), 0, 1,[225, 0, 0], 2, cv2.LINE_AA)
                #cv2.imwrite('/home/jason/opticalflow-X/images/{}.png'.format(info1[3]),info1[0])
                #cv2.imshow('info1[0]',info1[0])
                #cv2.waitKey(1)
                publish_image(debug_img_publish,info1[3],info1[0])
                data_copy_temp = info1[4]
            else:
                data_object_i = 0
                for res in info1[1]:
                    data_copy_temp[data_object_i].camInfo.u =  int(round(res['box'][0]*size_scale[0])) if res['box'][0] >=0 else 0
                    data_copy_temp[data_object_i].camInfo.v = int(round(res['box'][1]*size_scale[1]))
                    data_copy_temp[data_object_i].camInfo.width = int(round((res['box'][2]-res['box'][0])*size_scale[2]))
                    data_copy_temp[data_object_i].camInfo.height = int(round((res['box'][3]-res['box'][1])*size_scale[3]))
                    #print('res[box]',res['box'],data_copy_temp[data_object_i].camInfo)
                    """
                    xyxy = np.array([data_copy_temp[data_object_i].camInfo.u,\
                                     data_copy_temp[data_object_i].camInfo.v,\
                                     data_copy_temp[data_object_i].camInfo.u+data_copy_temp[data_object_i].camInfo.width,\
                                     data_copy_temp[data_object_i].camInfo.v+data_copy_temp[data_object_i].camInfo.height])#res['box']
                    info1[0] = cv2.resize(info1[0],(1920, 1208), interpolation=cv2.INTER_CUBIC)
                    """
                    xyxy = res['box']
                                                      
                    cls = res['category']
                    
                    plot_one_box(xyxy ,info1[0], label=str(int(cls)), color=[0, 225, 0])
                    data_object_i += 1
                cv2.putText(info1[0], "{}".format(info1[2]), (10, 30), 0, 1,[0, 225, 0], 2, cv2.LINE_AA)
                #cv2.imwrite('/home/jason/opticalflow-X/images/{}.png'.format(info1[3]),info1[0])		                
                #cv2.imshow('info1[0]',info1[0])
                #cv2.waitKey(1)
                publish_image(debug_img_publish,info1[3],info1[0])
                #print('data_copy_temp',data_copy_temp)
                publish_bbox(tracking_bbox_publish,info1[3],data_copy_temp)
            full_image_bbox_info.pop(0)

def ros_publisher_debug():
    debug_img_publish=rospy.Publisher('/cam/front_bottom_60/optical_flow',Image,queue_size=1)
    tracking_bbox_publish=rospy.Publisher('/optical_flow/front_bottom_60',DetectedObjectArray,queue_size=1)    
    debug_img(debug_img_publish,tracking_bbox_publish)
            
            
def publish_image(image_pubulish,time_stamp,imgdata):
    img_msg = bridge.cv2_to_imgmsg(imgdata, "bgr8")
    header = Header(stamp=time_stamp)
    header.frame_id = 'lidar'
    img_msg.header =  header
    image_pubulish.publish(img_msg)             
            
            
def publish_bbox(tracking_bbox_publish,time_stamp,data_copy):
    bbox_temp = DetectedObjectArray()
    header = Header(stamp=time_stamp)
    header.frame_id = 'lidar'
    bbox_temp.header=header
    bbox_temp.objects = data_copy
    tracking_bbox_publish.publish(bbox_temp)          
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

img_time_stamp =[]

def img_subscriber():
    while not rospy.is_shutdown():
        img = rospy.wait_for_message('/cam/front_bottom_60', Image)
        img_time_stamp.append(img.header.stamp)
        
yolo_time_stamp = []

def yolo_subscriber():
    while not rospy.is_shutdown():
        string_msg = rospy.wait_for_message('/cam_obj/front_bottom_60', DetectedObjectArray)    
        yolo_time_stamp.append(string_msg.header.stamp)
        
        
        
def detect():
    # Initialize model
    op = optical()
    global colors
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(100)]

    width = 608
    height = 384

    counter = 0
    frame_id = 0
    time_count = 0
    yolo_count = 0
    optical_count = 0
    
    frame_id = 0
     
    img_thread = threading.Thread(target = ros_subscriber_image)
    yolo_thread = threading.Thread(target = ros_subscriber_yolo)
    yolo_timestamp_thread = threading.Thread(target = ros_subscriber_yolo_timestamp)
    debug_thread = threading.Thread(target = ros_publisher_debug)
    img_thread.start()
    yolo_thread.start()
    yolo_timestamp_thread.start()
    debug_thread.start()
    temp_img_len = 0
    global img_timestamp
    img1 = None
    frame_counter = 0
    global img_list
    global res_list
    global full_image_bbox_info
    global data_copy
    global keep_yolo_result
    global keep_data_object
    opt_len = 0
    y_len = 0
    first_yolo = False
    yolo_detected = False
    
    raw_img_size = np.array([1920 ,1208 ,1920 ,1208])
    yolo_img_size = np.array([608, 384 ,608, 384])
    size_scale = np.true_divide(yolo_img_size,raw_img_size)      
    
    while not rospy.is_shutdown():
        if not first_yolo:
            if yolo_timestamp_len > 0:
                print('yolo_timestamp_len',yolo_timestamp_len,keep_yolo_time,yolo_bbox_time)
                if keep_yolo_time[0] in yolo_bbox_time:
                    first_yolo = True
                    yolo_detected = True
                    yolo_index = img_timestamp.index(keep_yolo_time[0])
                    img1 = bridge.imgmsg_to_cv2(img_list[yolo_index][0], "bgr8")
                    res_list = []
                    for obj in keep_data_object[0]:
                        scale_bbox_info = np.array([int(obj.camInfo.u),\
                                                    int(obj.camInfo.v),\
                                                    int(obj.camInfo.u)+int(obj.camInfo.width),\
                                                    int(obj.camInfo.v)+int(obj.camInfo.height)])
                        scale_bbox_info = scale_bbox_info*size_scale
                        res = {
                                'box': [int(scale_bbox_info[0]), int(scale_bbox_info[1]),\
                                        int(scale_bbox_info[2]), int(scale_bbox_info[3])],
                                'category': int(obj.classId),
                                'confidence': float(obj.camInfo.prob)
                              }
                        res_list.append(res)  
                    full_image_bbox_info.append([img1,res_list,"YOLO",keep_yolo_time[0],keep_data_object[0]])
                    img_timestamp = img_timestamp[yolo_index:]
                    img_list = img_list[yolo_index:]
                    keep_yolo_time.pop(0)
                    yolo_bbox_time.pop(0)
                    keep_data_object.pop(0)
                    frame_counter = 1
                    y_len +=1

        else:
            if len(keep_yolo_time) >0:
                #optical
                #print('keep_yolo_time[0]',img_timestamp[frame_counter],keep_yolo_time[0],yolo_bbox_time)
                if img_timestamp[frame_counter] < keep_yolo_time[0]:
                    print('optical')
                    
                    img2 = bridge.imgmsg_to_cv2(img_list[frame_counter][0], "bgr8")
                    res_list,next_yolo = op(image_current=img1, image_next=img2, boundingbox_list=res_list)
                    full_image_bbox_info.append([img2,res_list,"Optical",img_timestamp[(frame_counter)]])
                    img1 = img2
                    frame_counter +=1
                    opt_len +=1
                    
                else:
                    #yolo
                    if len(yolo_bbox_time)>0 :
                        if keep_yolo_time[0] in yolo_bbox_time:
                            frame_counter = 0
                            yolo_result_index = yolo_bbox_time.index(keep_yolo_time[0])
                            print('yolo_result_index',yolo_result_index)
                            yolo_index = img_timestamp.index(keep_yolo_time[0])
                            print('yolo',img_timestamp[yolo_index]==keep_yolo_time[0])
                            img_list = img_list[yolo_index:]
                            img_timestamp = img_timestamp[yolo_index:]


                            img1 = bridge.imgmsg_to_cv2(img_list[frame_counter][0], "bgr8")
                            #img1 =  cv2.resize(img1,(1920,1208),interpolation=cv2.INTER_CUBIC)
                            res_list = []
                            for obj in keep_data_object[0]:
                                scale_bbox_info = np.array([int(obj.camInfo.u),\
                                                            int(obj.camInfo.v),\
                                                            int(obj.camInfo.u)+int(obj.camInfo.width),\
                                                            int(obj.camInfo.v)+int(obj.camInfo.height)])
                                scale_bbox_info = scale_bbox_info*size_scale
                                res = {
                                        'box': [int(scale_bbox_info[0]), int(scale_bbox_info[1]),\
                                                int(scale_bbox_info[2]), int(scale_bbox_info[3])],
                                        'category': int(obj.classId),
                                        'confidence': float(obj.camInfo.prob)
                                      }
                                res_list.append(res)  


                            full_image_bbox_info.append([img1,res_list,"YOLO",keep_yolo_time[0],keep_data_object[0]])
                            frame_counter += 1                        
                            y_len +=1
                            keep_yolo_time.pop(0)
                            keep_data_object.pop(0)




    print('yolo_len',yolo_len)
    print('img_len',img_len)
    print('opt_len',opt_len)
    print('y_len',y_len)
            
            
            
    img_thread.join()
    yolo_thread.join()
    yolo_timestamp_thread.join()
    debug_thread.join()

            
        


if __name__ == '__main__':
    detect()
    #rospy.spin()
