#!/usr/bin/env python2

#from utils.utils import *
import cv2
import copy
import datetime
import json
import time
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt


class ObjectDetectionsWithOpticalFlow:
    def __init__(self):
        cv2.ocl.setUseOpenCL(False)
        t = time.time()
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        #self.ori_rank = []
        #print('__init__',time.time()-t)

    def _img_resize(self):
        t = time.time()
        self.img1 = cv2.resize(self.img1, self.img_size)
        self.img2 = cv2.resize(self.img2, self.img_size)
        #print('_img_resize',time.time()-t)

    def _rescale(self, img0_shape, img1_shape, points_list):
        t = time.time()
        # Rescale x1, y1, x2, y2 from 416 to image size
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = [(img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2]*2 #[width padding, height padding]
        p1 = [list(map(lambda x,y:x/y, points, [gain]*4)) for points in [list(map(lambda x,y:x-y, points, pad)) for points in points_list]]
        #print('_rescale',time.time()-t)
        return p1

    def _bbox_group(self):
        t = time.time()
        normal_bbox = list()
        self.ori_rank = []
        self.extreme_small_box = list()
        for bbox in self.bbox_list:
            if ((bbox[2] - bbox[0]) <= self.crop_thresh) and ((bbox[3] - bbox[1]) <= self.crop_thresh):
                self.extreme_small_box.append(bbox)
                self.ori_rank.append(1)
            else:
                normal_bbox.append(bbox)
                self.ori_rank.append(0)
        self.bbox_list = normal_bbox
        #print('_bbox_group',time.time()-t)
        
    def _grayscale(self):
        t = time.time()
        self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        #print('_grayscale',time.time()-t)

    def _get_tracking_points(self, boxes):
        t = time.time()
        p0 = [[np.float32(random.randint(int(box[0] + (box[2] - box[0]) / 3),
                                         int(box[2] - (box[2] - box[0]) / 3))),\
               np.float32(random.randint(int(box[1] + (box[3] - box[1]) / 3), 
                                         int(box[3] - (box[3] - box[1]) / 3)))]\
                     for box in boxes for i in range(self.num_of_tracking_point)]
        #print('_get_tracking_points',time.time()-t)
        return p0

    def _calcOpticalFlowPyrLK(self, img1, img2, p0):   
        t = time.time()
        try:
            p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **self.lk_params)
        except cv2.error as e:
            raise e
        #print('_calcOpticalFlowPyrLK',time.time()-t)
        return p1

    def _optical_diffence(self, p0, p1, bbox_list):
        t = time.time()
        move = (p1 - p0)
        if (move>50).any():
            self.next_yolo = True
        move = move.reshape(-1, 2)
        x_move_median = np.nanmean(move[:, 0].reshape(-1, self.num_of_tracking_point), axis=1)
        y_move_median = np.nanmean(move[:, 1].reshape(-1, self.num_of_tracking_point), axis=1)
        move_median = np.asarray([x_move_median, y_move_median] * 2).T
        bbox_list_new = np.array(np.array(bbox_list).tolist()).astype(x_move_median.dtype)
        ind = np.arange(4)
        bbox_list_new[:, ind] = bbox_list_new[:, ind] + move_median
        bbox_list_new = bbox_list_new.tolist()
        #print('_optical_diffence',time.time()-t)
        return bbox_list_new
        
    def _box_to_int(self,bbox_list_new):
        t = time.time()
        int_box_list = [list(map(int, np.round(box))) for box in bbox_list_new]
        #print('_box_to_int',time.time()-t)
        return int_box_list

    def _extreme_small_crop(self):
        t = time.time()
        times = 2
        times_minus = (times - 1/2)
        times_plus  = (times + 1/2)

        mean_tmp = [[int((small_box[0]+small_box[2])/2),\
                     int((small_box[1]+small_box[3])/2),\
                     int(small_box[2]-small_box[0]),\
                     int(small_box[3]-small_box[1])]\
                     for small_box in self.extreme_small_box]        

        img1_crop = [self.img1[(mean_tmp[i][1]-mean_tmp[i][3]*times):(mean_tmp[i][1]+mean_tmp[i][3]*times),\
                                  (mean_tmp[i][0]-mean_tmp[i][2]*times) :(mean_tmp[i][0]+mean_tmp[i][2]*times)]
                     for i in range(len(self.extreme_small_box))]
        
        img2_crop = [self.img2[(mean_tmp[i][1]-mean_tmp[i][3]*times):(mean_tmp[i][1]+mean_tmp[i][3]*times),\
                                  (mean_tmp[i][0]-mean_tmp[i][2]*times) :(mean_tmp[i][0]+mean_tmp[i][2]*times)]
                     for i in range(len(self.extreme_small_box))]
        
        #for i in range(len(self.extreme_small_box)):
        #    print(img1_crop[i].shape,img2_crop[i].shape)
        #    cv2.imwrite('/home/lipei/20191120/check{}0.png'.format(i),img1_crop[i])
        #    cv2.imwrite('/home/lipei/20191120/check{}1.png'.format(i),img2_crop[i])
        
        box_small = [[ int(mean_tmp[i][2]*times_minus),\
                       int(mean_tmp[i][3]*times_minus),\
                       int(mean_tmp[i][2]*times_plus),\
                       int(mean_tmp[i][3]*times_plus)]
                     for i in range(len(self.extreme_small_box))]
             
        p0_crop = np.array(np.expand_dims(self._get_tracking_points(box_small), axis=1))
        p1_crop = np.array([self._calcOpticalFlowPyrLK(img1_crop[i], img2_crop[i], p0_crop[i*3:((i+1)*3),:])
                            for i in range(int(len(p0_crop)/3))]).reshape(-1,1,2)

        new_bbox_crop = self._optical_diffence(p0_crop, p1_crop, box_small)
        new_extreme_small_box = [[mean_tmp[i][0]-mean_tmp[i][2]*times+new_bbox_crop[i][0],\
                                  mean_tmp[i][1]-mean_tmp[i][3]*times+new_bbox_crop[i][1],\
                                  mean_tmp[i][0]-mean_tmp[i][2]*times+new_bbox_crop[i][2],\
                                  mean_tmp[i][1]-mean_tmp[i][3]*times+new_bbox_crop[i][3]] for i in range(len(new_bbox_crop))]
        #print('_extreme_small_crop',time.time()-t,len(new_extreme_small_box))
        return new_extreme_small_box

    def _normal_run(self): 
        t = time.time()
        normal_p0 = np.array(np.expand_dims(self._get_tracking_points(self.bbox_list), axis=1))
        normal_p1 = self._calcOpticalFlowPyrLK(self.img1, self.img2, normal_p0)
        new_normal_box = self._optical_diffence(normal_p0, normal_p1, self.bbox_list)
        return new_normal_box
        
    def __call__(self, image_current, image_next, boundingbox_list, num_of_tracking_point=10, resize=None, crop_threshold=None):
        tt = time.time()

        self.next_yolo = False
        self.img1 = image_current
        self.img1_shape = self.img1.shape[0:2]
        self.img2 = image_next
        self.bbox_list = [det['box'] for det in boundingbox_list]
        if not self.bbox_list:
            return [],self.next_yolo
        self.img_size = resize
        self.crop_thresh = crop_threshold
        self.num_of_tracking_point = num_of_tracking_point
        self.output = copy.copy(boundingbox_list)
        self._grayscale()
        self.interger = True
        

        if self.crop_thresh != None:
            self._bbox_group()
            new_extreme_small_box = self._extreme_small_crop()
            if self.interger == True:
                new_extreme_small_box = self._box_to_int(new_extreme_small_box)
         
        if self.img_size != None:
            self.bbox_list = self._rescale(self.img_size,self.img1_shape, self.bbox_list)
            self._img_resize() 
            
        if self.bbox_list: # bbox_list is not empty
            new_normal_box = self._normal_run()
            #print('new_normal_box',new_normal_box)
            if self.img_size != None:
                new_normal_box = self._rescale(self.img1_shape,self.img_size, new_normal_box)
            if self.interger == True:
                new_normal_box = self._box_to_int(new_normal_box)
        
        if 'ori_rank' not in globals():    
            i = 0
            for nnbox in new_normal_box:
                self.output[i]['box'] = nnbox
                i += 1
        else:
            nor_num = 0
            exe_num = 0
            rr = 0
            for ori_r in self.ori_rank:
                print(rr)
                if ori_r == 0:
                    self.output[rr]['box'] = new_normal_box[nor_num]
                    nor_num += 1
                else:
                    self.output[rr]['box'] = new_extreme_small_box[exe_num]
                    exe_num += 1
                rr +=1
        print('__call__', time.time()-tt)
        return self.output,self.next_yolo





