#!/usr/bin/env python2
import psutil
p = psutil.Process()
p.cpu_affinity()
p.cpu_affinity([1,2,3])
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
from opticalflow_motion import ObjectDetectionsWithOpticalFlow as optical
import sys 
from msgs.msg import DetectedObjectArray


class ros_detect_opticalflow:
    def __init__(self):
        self.bridge = CvBridge()
        self.same_times = 0
        self.img_len = 0
        self.temp_img_len = None
        self.last_temp_img_len = None
        self.yolo_len = 0
        self.temp_yolo_len = None
        self.frame_counter = 0
        self.yolo_timestamp_len = 0
        self.temp_img_len = None
        self.temp_yolo_len = None
        self.first_yolo = False
        self.img_timestamp = []
        self.img_list = []
        self.keep_yolo_time = []
        self.yolo_bbox_time = []
        self.keep_data_object = []
        rospy.init_node('listener', anonymous=True)
        self.inputTopic = rospy.get_param("~topic")

        
    def ros_subscriber_image(self):
        #print('self.inputTopic',self.inputTopic)
        rospy.Subscriber('/cam/{}'.format(self.inputTopic), Image, self.img_callback)
        rospy.spin()

    def img_callback(self,data): 
        #print('self.img_len',self.img_timestamp)
        self.img_list.append([data,data.header.stamp]) #bridge.imgmsg_to_cv2(data, "bgr8")) 
        self.img_timestamp.append(data.header.stamp)
        self.img_len += 1


    def ros_subscriber_yolo_timestamp(self):    
        rospy.Subscriber('/cam_obj/{}/time_info'.format(self.inputTopic), Header,\
                         self.yolo_timestamp_callback)
        rospy.spin()

    def yolo_timestamp_callback(self,data):
        #print('yolo_timestamp_len',data.stamp)
        self.yolo_timestamp_len += 1
        self.keep_yolo_time.append(data.stamp)    

    def ros_subscriber_yolo(self):
        rospy.Subscriber('/cam_obj/{}'.format(self.inputTopic), DetectedObjectArray, self.yolo_callback)
        rospy.spin()

    def yolo_callback(self,data):
        #print('yolo_bbox_time',self.yolo_bbox_time)
        self.yolo_len += 1
        self.yolo_bbox_time.append(data.header.stamp)    
        self.keep_data_object.append(data.objects)
    
    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
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
            
            
    def debug_img(self, debug_img_publish, full_image_info):
        raw_img_size = np.array([1920 ,1208 ,1920 ,1208])
        yolo_img_size = np.array([608, 384 ,608, 384])
        size_scale = np.true_divide(raw_img_size,yolo_img_size)      
        rate = rospy.Rate(20)
        info1 = full_image_info
        if info1[2] == "YOLO":
            for res in info1[1]:
                xyxy = res['box']
                cls = res['category']
                self.plot_one_box(xyxy, info1[0], label=str(int(cls)), color=[225, 0, 0])
            cv2.putText(info1[0], "{}".format(info1[2]), (10, 30), 0, 1,[225, 0, 0], 2, cv2.LINE_AA)
            self.publish_image(debug_img_publish,info1[3],info1[0])
        else:
            data_object_i = 0
            for res in info1[1]:
                xyxy = res['box']
                cls = res['category']
                self.plot_one_box(xyxy ,info1[0], label=str(int(cls)), color=[0, 225, 0])
                data_object_i += 1
            cv2.putText(info1[0], "{}".format(info1[2]), (10, 30), 0, 1,[0, 225, 0], 2, cv2.LINE_AA)
            self.publish_image(debug_img_publish,info1[3],info1[0])
    
    def tracking_bbox(self, tracking_bbox_publish, full_bbox_info):
        raw_img_size = np.array([1920 ,1208 ,1920 ,1208])
        yolo_img_size = np.array([608, 384 ,608, 384])
        size_scale = np.true_divide(raw_img_size,yolo_img_size)      
        rate = rospy.Rate(20)
        info1 = full_bbox_info
        if info1[1] == "YOLO":
            self.publish_bbox(tracking_bbox_publish,info1[2],info1[3])
            self.data_copy_temp = info1[3]
        else:
            data_object_i = 0
            for res in info1[0]:
                self.data_copy_temp[data_object_i].camInfo[0].u =  \
                   int(round(res['box'][0]*size_scale[0])) if res['box'][0] >=0 else 0
                self.data_copy_temp[data_object_i].camInfo[0].v = \
                   int(round(res['box'][1]*size_scale[1])) if res['box'][1] >=0 else 0
                self.data_copy_temp[data_object_i].camInfo[0].width = \
                   int(round((res['box'][2]-res['box'][0])*size_scale[2]))
                self.data_copy_temp[data_object_i].camInfo[0].height = \
                   int(round((res['box'][3]-res['box'][1])*size_scale[3]))
                data_object_i += 1

            self.publish_bbox(tracking_bbox_publish,info1[2],self.data_copy_temp)


                
    def ros_publisher_debug_img(self, full_image_info):
        debug_img_publish=rospy.Publisher('/cam/{}/optical_flow'.format(self.inputTopic),Image,queue_size=1)
        self.debug_img(debug_img_publish, full_image_info)

    def ros_publisher_optical_bbox(self, full_bbox_info):
        tracking_bbox_publish=rospy.Publisher('/optical_flow/{}'.format(self.inputTopic),\
                                              DetectedObjectArray,queue_size=1)   
        self.tracking_bbox(tracking_bbox_publish, full_bbox_info)                
        
    def publish_image(self, image_pubulish,time_stamp,imgdata):
        img_msg = self.bridge.cv2_to_imgmsg(imgdata, "bgr8")
        header = Header(stamp=time_stamp)
        header.frame_id = 'lidar'
        img_msg.header =  header
        image_pubulish.publish(img_msg)             

    def publish_bbox(self, tracking_bbox_publish,time_stamp,data_copy):
        bbox_temp = DetectedObjectArray()
        header = Header(stamp=time_stamp)
        header.frame_id = 'lidar'
        bbox_temp.header=header
        bbox_temp.objects = data_copy
        tracking_bbox_publish.publish(bbox_temp)    
        
    def check_status_every_sec(self):
        #while not rospy.is_shutdown():
            #print('hello lipei',self.same_times,self.last_temp_img_len ,self.temp_img_len,self.img_len)
            if self.temp_img_len == self.img_len:
                if self.last_temp_img_len == self.temp_img_len:
                    self.same_times += 1 
                if self.same_times > 2:
                    self.same_times = 0
                    self.img_len = 0
                    self.last_temp_img_len = None
                    self.temp_img_len = None
                    self.temp_yolo_len = None
                    self.yolo_len = 0
                    self.frame_counter = 0
                    self.yolo_timestamp_len = 0
                    self.first_yolo = False
                    self.img_timestamp = []
                    self.img_list = []
                    self.keep_yolo_time = []
                    self.yolo_bbox_time = []
                    self.keep_data_object = []
                    self.first_yolo = False
                    self.yolo_detected = False
            else:
                self.same_times = 0
                self.temp_img_len =self.img_len
            self.last_temp_img_len = self.temp_img_len
            #time.sleep(1)


    def run(self):
        # Initialize model
        op = optical()
        img_thread = threading.Thread(target = self.ros_subscriber_image)
        yolo_thread = threading.Thread(target = self.ros_subscriber_yolo)
        yolo_timestamp_thread = threading.Thread(target = self.ros_subscriber_yolo_timestamp)
        
        
        img_thread.start()
        yolo_thread.start()
        yolo_timestamp_thread.start()


        opt_len = 0
        y_len = 0
        self.first_yolo = False
        self.yolo_detected = False

        raw_img_size = np.array([1920 ,1208 ,1920 ,1208])
        yolo_img_size = np.array([608, 384 ,608, 384])
        size_scale = np.true_divide(yolo_img_size,raw_img_size)      
        rate = rospy.Rate(20)          
        while not rospy.is_shutdown():
            self.check_status_every_sec()
            if not self.first_yolo:
                if len(self.keep_yolo_time) >0 and len(self.yolo_bbox_time) >0:
                    #print('first',self.keep_yolo_time,self.yolo_bbox_time,self.img_timestamp)
                    duplicate_yolo_img_time = np.nonzero(np.isin(self.keep_yolo_time, self.img_timestamp))[0]
                    if len(duplicate_yolo_img_time)>0:
                       self.keep_yolo_time = self.keep_yolo_time[duplicate_yolo_img_time[0]:]
                       duplicate_yolo_time = np.nonzero(np.isin(self.keep_yolo_time, self.yolo_bbox_time))[0] 
                       #print('duplicate_yolo_time',duplicate_yolo_time)
                       if len(duplicate_yolo_time) >0:
                            self.temp_img_len = self.img_len
                            self.temp_yolo_len = self.yolo_len
                            self.first_yolo = True
                            self.yolo_detected = True
                            yolo_result_index = self.yolo_bbox_time.index(\
                                                       self.keep_yolo_time[duplicate_yolo_time[0]])
                            yolo_index = self.img_timestamp.index(self.keep_yolo_time[duplicate_yolo_time[0]])
                            img1 = self.bridge.imgmsg_to_cv2(self.img_list[yolo_index][0], "bgr8")
                            res_list = []
                            for obj in self.keep_data_object[yolo_result_index]:
                                scale_bbox_info = np.array([int(obj.camInfo[0].u),\
                                                            int(obj.camInfo[0].v),\
                                                            int(obj.camInfo[0].u)+int(obj.camInfo[0].width),\
                                                            int(obj.camInfo[0].v)+int(obj.camInfo[0].height)])
                                scale_bbox_info = scale_bbox_info*size_scale
                                res = {
                                        'box': [int(scale_bbox_info[0]), int(scale_bbox_info[1]),\
                                                int(scale_bbox_info[2]), int(scale_bbox_info[3])],
                                        'category': int(obj.classId),
                                        'confidence': float(obj.camInfo[0].prob)
                                      }
                                res_list.append(res)  
  
                            full_image_info = [img1,res_list,"YOLO",\
                                               self.keep_yolo_time[duplicate_yolo_time[0]]]
                            self.ros_publisher_debug_img(full_image_info)
                            full_bbox_info=[res_list,"YOLO",\
                                            self.keep_yolo_time[duplicate_yolo_time[0]],\
                                            self.keep_data_object[yolo_result_index]]
                            self.ros_publisher_optical_bbox(full_bbox_info)
                            self.img_timestamp = self.img_timestamp[yolo_index+1:]
                            self.img_list = self.img_list[yolo_index+1:]
                            self.yolo_bbox_time = self.yolo_bbox_time[(yolo_result_index+1):]
                            self.keep_data_object = self.keep_data_object[(yolo_result_index+1):]
                            self.keep_yolo_time = self.keep_yolo_time[duplicate_yolo_time[0]+1:]
                            self.frame_counter = 0
                            y_len +=1

            else:
                if len(self.keep_yolo_time) >0 and len(self.img_timestamp)>0:
                    #optical
                    if self.img_timestamp[self.frame_counter] < self.keep_yolo_time[0]:
                        #print('opt')
                        self.temp_img_len = self.img_len
                        self.temp_yolo_len = self.yolo_len
                        img2 = self.bridge.imgmsg_to_cv2(self.img_list[self.frame_counter][0], "bgr8")
                        a_t = time.time()
                        res_list,_ = op(image_current=img1, image_next=img2, boundingbox_list=res_list)
                        full_image_info = [img2,res_list,"Optical",\
                                                     self.img_timestamp[(self.frame_counter)]]
                        self.ros_publisher_debug_img(full_image_info)
                        full_bbox_info = [res_list,"Optical",\
                                                    self.img_timestamp[(self.frame_counter)]]
                        self.ros_publisher_optical_bbox(full_bbox_info)
                        img1 = img2
                        self.frame_counter +=1
                        opt_len +=1

                    else:
                        #yolo
                        if len(self.yolo_bbox_time)>0 and len(self.img_timestamp) >0:
                            if self.keep_yolo_time[0] in self.yolo_bbox_time:
                                if self.keep_yolo_time[0] not in self.img_timestamp:
                                    self.keep_yolo_time = self.keep_yolo_time[1:]
                                else:
                                    #print('yolo')
                                    self.temp_img_len = self.img_len
                                    self.temp_yolo_len = self.yolo_len
                                    self.frame_counter = 0
                                    yolo_result_index = self.yolo_bbox_time.index(self.keep_yolo_time[0])
                                    yolo_index = self.img_timestamp.index(self.keep_yolo_time[0])
                                    self.img_list = self.img_list[yolo_index:]
                                    self.img_timestamp = self.img_timestamp[yolo_index+1:]

                                    img1 = self.bridge.imgmsg_to_cv2(self.img_list[self.frame_counter][0], "bgr8")
                                    self.img_list = self.img_list[1:]

                                    res_list = []
                                    for obj in self.keep_data_object[yolo_result_index]:
                                        scale_bbox_info = np.array([int(obj.camInfo[0].u),\
                                                                    int(obj.camInfo[0].v),\
                                                                    int(obj.camInfo[0].u)+int(obj.camInfo[0].width),\
                                                                    int(obj.camInfo[0].v)+int(obj.camInfo[0].height)])
                                        scale_bbox_info = scale_bbox_info*size_scale
                                        res = {
                                                'box': [int(scale_bbox_info[0]), int(scale_bbox_info[1]),\
                                                        int(scale_bbox_info[2]), int(scale_bbox_info[3])],
                                                'category': int(obj.classId),
                                                'confidence': float(obj.camInfo[0].prob)
                                              }
                                        res_list.append(res)  

                                    full_image_info = [img1,res_list,"YOLO",self.keep_yolo_time[0]]
                                    self.ros_publisher_debug_img(full_image_info)
                                    full_bbox_info = [res_list,"YOLO",self.keep_yolo_time[0],\
                                                                self.keep_data_object[yolo_result_index]]       
                                    self.ros_publisher_optical_bbox(full_bbox_info)
                                    y_len +=1
                                    self.yolo_bbox_time = self.yolo_bbox_time[(yolo_result_index+1):]
                                    self.keep_data_object = self.keep_data_object[(yolo_result_index+1):]
                                    self.keep_yolo_time = self.keep_yolo_time[1:]

            rate.sleep()

        img_thread.join()
        yolo_thread.join()
        yolo_timestamp_thread.join()

        
        #print('yolo_len',self.yolo_len)
        #print('img_len',self.img_len)
        #print('opt_len',opt_len)
        #print('y_len',y_len)



        

            



if __name__ == '__main__':
    A = ros_detect_opticalflow()
    A.run()
    
