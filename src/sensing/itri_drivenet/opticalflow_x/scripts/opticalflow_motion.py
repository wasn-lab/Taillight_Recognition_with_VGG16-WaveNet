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







class ObjectDetectionsWithMotionvector:
    def __init__(self):
        print('init')

    def _json_load_to_array(self, json_path):
        # json_load_to_array_st = time.time()
        input_file = open(json_path)
        json_array = np.asarray(list(map(lambda x: [x['src_x']-1, x['src_y']-1, x['dx'], x['dy']],
                                         json.load(input_file))))
        self.mv_src = json_array[:, :2]
        self.mv_src_copy = copy.deepcopy(self.mv_src)
        self.mv_d = json_array[:, 2:4]
        self.mv_d_copy = copy.deepcopy(self.mv_d)
        self.mvs = json_array
        # self.json_load_to_array_t = time.time()-json_load_to_array_st
        # print ('hi0',time.time()-st)

    def _remove_bbox_mv(self):
        # remove_bbox_mv_st = time.time()
        temp_mv_src = copy.deepcopy(self.mv_src)
        temp_mv_d = copy.deepcopy(self.mv_d)
        new_temp_index = np.ones(len(temp_mv_src), bool)
        for bbox in self.bbox_list:
            temp_index = np.where((temp_mv_src[:, 0] > bbox[0]) & (temp_mv_src[:, 0] < bbox[2]) &
                                  (temp_mv_src[:, 1] > bbox[1]) & (temp_mv_src[:, 1] < bbox[3]))[0]
            new_temp_index[temp_index] = False
        self.mv_src = temp_mv_src[new_temp_index]
        self.mv_d = temp_mv_d[new_temp_index]
        # self.remove_bbox_mv_t = time.time()-remove_bbox_mv_st

    def _cal_residual(self):
        cal_residual_st = time.time()
        for n in range(len(self.four_direction_lim_update)):
            temp_direction_lim = self.four_direction_lim_update[n]
            temp_region_index = np.where((self.mv_src[:, 0] > temp_direction_lim[0]) &
                                         (self.mv_src[:, 0] < temp_direction_lim[1]) &
                                         (self.mv_src[:, 1] > temp_direction_lim[2]) &
                                         (self.mv_src[:, 1] < temp_direction_lim[3]))
            bg_src = (self.mv_src[temp_region_index])
            bg_d = (self.mv_d[temp_region_index])
            self.residual.extend([np.mean(((self.dir_m_c[n][0]*bg_src[:, 0]+self.dir_m_c[n][1])-bg_d[:, 0])**2 +
                                          ((self.dir_m_c[n][2]*bg_src[:, 1]+self.dir_m_c[n][3])-bg_d[:, 1])**2)])
        # self.cal_residual_t = time.time()-cal_residual_st
        # print('self.cal_residual_t',self.cal_residual_t)

    def _global_mv_four_quadrant(self):
        # global_mv_four_quadrant_st = time.time()
        if self.x_lim is None:
            self.x_lim = self.img_width

        if self.y_lim is None:
            self.y_lim = self.img_height

        if self.direction_length == 4:
            self.four_direction_lim = [[0, (self.x_lim/2), 0, (self.y_lim/2)],
                                       [(self.x_lim/2), (self.x_lim), 0, (self.y_lim/2)],
                                       [0, (self.x_lim/2), (self.y_lim/2), (self.y_lim)],
                                       [(self.x_lim/2), (self.x_lim), (self.y_lim/2), (self.y_lim)]]
        elif self.direction_length == 2:
            self.four_direction_lim = [
                [0, (self.x_lim), (self.y_lim)/2, (self.y_lim)]]

        self.dir_m_c = []
        self.residual = []
        self.four_direction_lim_update = np.array(self.four_direction_lim)[
            np.array(self.direction_area)].tolist()

        for n in range(len(self.four_direction_lim_update)):
            temp_direction_lim = self.four_direction_lim_update[n]
            temp_region_index = np.where((self.mv_src[:, 0] > temp_direction_lim[0]) &
                                         (self.mv_src[:, 0] < temp_direction_lim[1]) &
                                         (self.mv_src[:, 1] > temp_direction_lim[2]) &
                                         (self.mv_src[:, 1] < temp_direction_lim[3]))
            temp_region_src = self.mv_src[temp_region_index]
            temp_region_d = self.mv_d[temp_region_index]

            temp_src_x = np.vstack(
                [temp_region_src[:, 0], np.ones(len(temp_region_src[:, 0]))]).T
            temp_x_coef, x_residual = np.linalg.lstsq(
                temp_src_x, temp_region_d[:, 0], rcond=None)[:2]
            temp_src_x_m, temp_src_x_c = temp_x_coef
             
            self.temp_src_x_m = temp_src_x_m
            self.temp_src_x_c = temp_src_x_c

            temp_src_y = np.vstack(
                [temp_region_src[:, 1], np.ones(len(temp_region_src[:, 1]))]).T
            temp_y_coef, y_residual = np.linalg.lstsq(
                temp_src_y, temp_region_d[:, 1], rcond=None)[:2]
            temp_src_y_m, temp_src_y_c = temp_y_coef

            self.temp_src_y_m = temp_src_y_m
            self.temp_src_y_c = temp_src_y_c
            self.residual.append(
                np.float((x_residual+y_residual)/len(temp_region_index[0])))
            self.dir_m_c.append(
                [temp_src_x_m, temp_src_x_c, temp_src_y_m, temp_src_y_c])

        #     plt.figure(1) 
        #     plt.subplot(5, 2, n * 2 + 1)
        #     plt.scatter(temp_region_src[:, 0], temp_region_d[:, 0], c='blue', alpha=0.75)
        #     plt.plot(temp_region_src[:, 0], temp_src_x_m * temp_region_src[:, 0] + temp_src_x_c, c='red', alpha=0.75)
        #     plt.subplot(5, 2, n * 2 + 2)
        #     plt.scatter(temp_region_src[:, 1], temp_region_d[:, 1], c='blue', alpha=0.75)
        #     plt.plot(temp_region_src[:, 1], temp_src_y_m * temp_region_src[:, 1] + temp_src_y_c, c='red', alpha=0.75)       
        # # self.global_mv_four_quadrant_t = time.time()-global_mv_four_quadrant_st
        # plt.draw()
        # plt.pause(1)
 
    def _check_bbox_region_x(self, bbox):
        if bbox[0] < (self.x_lim/2) and bbox[2] > (self.x_lim/2):
            return [[bbox[0]+self.half_searchwindow,
                     bbox[1]+self.half_searchwindow,
                     int(self.x_lim/2),
                     bbox[3]-self.half_searchwindow],
                    [int(self.x_lim/2),
                     bbox[1]+self.half_searchwindow,
                     bbox[2]-self.half_searchwindow,
                     bbox[3]-self.half_searchwindow]]
        else:
            return [[bbox[0]+self.half_searchwindow,
                     bbox[1]+self.half_searchwindow,
                     bbox[2]-self.half_searchwindow,
                     bbox[3]-self.half_searchwindow]]

    def _remain_bbox_mv_vector(self, bbox, xy_linear, residual):
        index = np.where((self.mv_src_copy[:, 0] > bbox[0]) & (self.mv_src_copy[:, 0] < bbox[2]) &
                         (self.mv_src_copy[:, 1] > bbox[1]) & (self.mv_src_copy[:, 1] < bbox[3]))[0]
        if len(index) == 0:
            return []

        intersection_index_and_redmvsindex = list(
            set(index) & set(self.remain_red_mvs_index))
        if len(intersection_index_and_redmvsindex)/len(index) >= 0.2:
            return intersection_index_and_redmvsindex

        x_linear_m, x_linear_c, y_linear_m, y_linear_c = xy_linear

        bbox_ori_mv_src_x = self.mv_src_copy[index, 0]
        bbox_ori_mv_d_x = self.mv_d_copy[index, 0]
        bbox_pred_mv_d_x = x_linear_m*bbox_ori_mv_src_x+x_linear_c

        bbox_ori_mv_src_y = self.mv_src_copy[index, 1]
        bbox_ori_mv_d_y = self.mv_d_copy[index, 1]
        bbox_pred_mv_d_y = y_linear_m*bbox_ori_mv_src_y+y_linear_c

        distance = (bbox_pred_mv_d_x-bbox_ori_mv_d_x)**2 + \
            (bbox_pred_mv_d_y-bbox_ori_mv_d_y)**2
        pixel_threshold = residual
        inverse_with_bg = np.where(distance >= pixel_threshold)[0]

        similar_with_bg = np.where(distance < pixel_threshold)[0]

        if len(inverse_with_bg) > len(similar_with_bg):
            self.mv_class_index += 1
            self.gmv_index.extend(index)
            return index[inverse_with_bg]
        else:
            self.mv_class_index += 0
            self.gmv_index.extend(index)
            return index[similar_with_bg]

    def _bbox_limit(self, bbox):
        bbox[0] = bbox[0] if bbox[0] >= 0 else 0
        bbox[1] = bbox[1] if bbox[1] >= 0 else 0
        bbox[2] = bbox[2] if bbox[2] < self.img_width else (self.img_width-1)
        bbox[3] = bbox[3] if bbox[3] < self.img_height else (self.img_height-1)
        return bbox

    def _into_group_new(self, bbox):
        if (np.round(bbox[2]-bbox[0]) < self.half_searchwindow or
                np.round(bbox[3]-bbox[1]) < self.half_searchwindow):
            if self.draw_mv_in_bbox:
                return [], []
            else:
                return []
        self.mv_class_index = 0
        if len(self.four_direction_lim_update) > 1:
            check_bbox = self._check_bbox_region_x(bbox)
            if len(check_bbox) == 1:
                if check_bbox[0][0] <= (self.x_lim/2) and check_bbox[0][2] <= (self.x_lim/2):
                    temp_index = self._remain_bbox_mv_vector(
                        check_bbox[0], self.dir_m_c[0], self.residual[0])
                elif check_bbox[0][0] >= (self.x_lim/2) and check_bbox[0][2] >= (self.x_lim/2):
                    temp_index = self._remain_bbox_mv_vector(
                        check_bbox[0], self.dir_m_c[1], self.residual[1])
                self.mv_class.append(self.mv_class_index)
            else:
                temp_index = []
                temp_index.extend(self._remain_bbox_mv_vector(
                    check_bbox[0], self.dir_m_c[0], self.residual[0]))
                temp_index.extend(self._remain_bbox_mv_vector(
                    check_bbox[1], self.dir_m_c[1], self.residual[1]))
                self.mv_class.append(self.mv_class_index)
        else:
            temp_index = self._remain_bbox_mv_vector(
                bbox, self.dir_m_c[0], self.residual[0])
            self.mv_class.append(self.mv_class_index)

        if len(temp_index) == 0:
            if self.draw_mv_in_bbox:
                return [], []
            else:
                return []

        if self.draw_mv_in_bbox:
            new_bbox, draw_mv = self._linear_regression(bbox, temp_index)
            return new_bbox, draw_mv
        else:
            new_bbox = self._linear_regression(bbox, temp_index)
            return new_bbox

    def _linear_regression(self, bbox, index):
        filter_src = self.mv_src_copy[index, :]
        filter_d = self.mv_d_copy[index, :]
        d_x = filter_d[:, 0]
        d_y = filter_d[:, 1]

        filter_src_x = np.vstack(
            [filter_src[:, 0], np.ones(len(filter_src[:, 0]))]).T
        x_model, x_resid = np.linalg.lstsq(filter_src_x, d_x, rcond=None)[:2]
        x_a, x_c = x_model

        filter_src_y = np.vstack(
            [filter_src[:, 1], np.ones(len(filter_src[:, 1]))]).T
        y_model, y_resid = np.linalg.lstsq(filter_src_y, d_y, rcond=None)[:2]
        y_a, y_c = y_model

        new_bbox = [bbox[0]+(bbox[0]*x_a+x_c),
                    bbox[1]+(bbox[1]*y_a+y_c),
                    bbox[2]+(bbox[2]*x_a+x_c),
                    bbox[3]+(bbox[3]*y_a+y_c)]
        self._bbox_limit(new_bbox)
        wh_ratio = (new_bbox[2]-new_bbox[0])/(new_bbox[3]-new_bbox[1])
        new_bbox.append(wh_ratio)

        # # plt.figure(2) 
        # plot9 = plt.subplot(5, 2, 9)
        # # plot9.set_ylim([-10, 10])
        # plot9.scatter(filter_src[:, 0], filter_d[:, 0], c='blue', alpha=0.75)
        # plot9.plot(filter_src[:, 0], x_a * filter_src[:, 0] + x_c, c='red', alpha=0.75)
        # plot10 = plt.subplot(5, 2, 10)
        # # plot10.set_ylim([-10, 10])
        # plot10.scatter(filter_src[:, 1], filter_d[:, 1], c='blue', alpha=0.75)
        # plot10.plot(filter_src[:, 1], y_a * filter_src[:, 1] + y_c, c='red', alpha=0.75)       
        # plt.draw()
        # plt.pause(1)
        # plot9.cla()
        # plot10.cla()
        
        if self.draw_mv_in_bbox:
            draw_mv = self._draw_mv(filter_src, filter_d)
            return new_bbox, draw_mv
        else:
            return new_bbox

    def _draw_mv(self, filter_src, filter_d, Q13_index=None):
        if Q13_index is not None:
            draw_mv_all = np.concatenate(
                (filter_src, filter_src+filter_d), 1).tolist()
            draw_mv_part = np.concatenate(
                (filter_src[Q13_index], filter_src[Q13_index]+filter_d[Q13_index]), 1).tolist()
            draw_mv = [draw_mv_all, draw_mv_part]
        else:
            draw_mv = np.concatenate(
                (filter_src, filter_src+filter_d), 1).tolist()
        return draw_mv

    def _get_bbox_mv(self):
        # bbox_tracking_st = time.time()
        if self.draw_mv_in_bbox:
            all_list = [self._into_group_new(bbox) for bbox in self.bbox_list]
            new_bbox = [ele1 for ele1, ele2 in all_list]
            self.draw_mv = [ele2 for ele1, ele2 in all_list]
        else:
            new_bbox = [self._into_group_new(bbox) for bbox in self.bbox_list]
        # self.bbox_tracking_t = time.time() - bbox_tracking_st
        return new_bbox

    def _box_to_int(self, new_bbox):
        int_box_list = [list(map(int, np.round(box))) for box in new_bbox]
        return int_box_list

    def _mv_map_generate(self):
        if self.residual[2] <= 1 or self.residual[3] <= 1:
            self.stopping = True
        else:
            self.stopping = False
        self.mv_x = np.zeros(
            (self.block_height, self.block_width, 1), np.uint8)
        if len(self.four_direction_lim) == 4:
            self.lim_left_d = [self.dir_m_c[0][0] * wid +
                               self.dir_m_c[0][1] for wid in self.lim_left_wid]
            self.lim_right_d = [self.dir_m_c[1][0] * wid +
                                self.dir_m_c[1][1] for wid in self.lim_right_wid]
        else:
            self.lim_left_d = [self.dir_m_c[0][0] * wid +
                               self.dir_m_c[0][1] for wid in self.lim_left_wid]
            self.lim_right_d = [self.dir_m_c[0][0] * wid +
                                self.dir_m_c[0][1] for wid in self.lim_right_wid]
        if (self.dual_carriageway):
            if (self.lim_left_d[3] < -4 and self.lim_right_d[3] < -4) or (self.lim_left_d[3] > 2 and self.lim_right_d[3] > 2):
                self.turning = True
                index = np.where((self.mvs[:, 0] < self.block_width * 16) & (self.mvs[:, 1] < self.block_height * 16) &
                                 (self.mvs[:, 1] > self.lim_height) & (self.mvs[:, 3] != 0) & (
                                ((self.mvs[:, 2] > 10) & (self.mvs[:, 0] < self.lim_left_wid[1]) & (self.lim_left_d[3] < 0)) |
                                ((self.mvs[:, 2] < -10) & (self.mvs[:, 0] > self.lim_right_wid[1]) & (self.lim_right_d[3] > 0))))
            else:
                self.turning = False
                index = np.where((self.mvs[:, 0] < self.block_width * 16) & (self.mvs[:, 1] < self.block_height * 16) &
                                 (self.mvs[:, 1] > self.lim_height) & (self.mvs[:, 3] != 0) &
                                 ((self.mvs[:, 2] > 5) & (self.mvs[:, 0] < self.lim_left_wid[3]) |
                                  (self.mvs[:, 2] < self.lim_left_d[0] - 10) & (self.mvs[:, 0] > self.lim_left_wid[0]) |
                                  (self.mvs[:, 2] < self.lim_left_d[1] - 10) & (self.mvs[:, 0] > self.lim_left_wid[1]) |
                                  (self.mvs[:, 2] < self.lim_left_d[2] - 6) & (self.mvs[:, 0] > self.lim_left_wid[2]) |
                                  (self.mvs[:, 2] < self.lim_left_d[3] - 6) & (self.mvs[:, 0] > self.lim_left_wid[3]) |
                                  (self.mvs[:, 2] < -5) & (self.mvs[:, 0] > self.lim_right_wid[3])))
        else:
            if (self.lim_left_d[3] < -4 and self.lim_right_d[3] < -4) or (self.lim_left_d[3] > 2 and self.lim_right_d[3] > 2):
                self.turning = True
                index = np.where((self.mvs[:, 0] < self.block_width * 16) & (self.mvs[:, 1] < self.block_height * 16) &
                                 (self.mvs[:, 1] > self.lim_height) & (self.mvs[:, 3] != 0) & (
                    ((
                        (self.mvs[:, 2] > 10) & (self.mvs[:, 0] < self.lim_left_wid[1]) |
                        (self.mvs[:, 2] < 0) & (
                            self.mvs[:, 0] > self.lim_right_wid[0])
                    ) & (self.lim_left_d[3] < 0)) |
                    ((
                        (self.mvs[:, 2] < -10) & (self.mvs[:, 0] > self.lim_right_wid[1]) |
                        (self.mvs[:, 2] > self.lim_left_d[0]) & (self.mvs[:, 0] < self.lim_left_wid[0]) |
                        (self.mvs[:, 2] > self.lim_left_d[1]) & (
                            self.mvs[:, 0] < self.lim_left_wid[1])
                    ) & (self.lim_right_d[3] > 0))
                ))
            else:
                self.turning = False
                if self.stopping:
                    index = np.where((self.mvs[:, 0] < self.block_width * 16) & (self.mvs[:, 1] < self.block_height * 16) &
                                     (self.mvs[:, 1] > self.lim_height) & (self.mvs[:, 3] == 0) &
                                     ((self.mvs[:, 2] > 5) | (self.mvs[:, 2] < -5)) & (self.mvs[:, 1] < self.img_height*3/4))
                else:
                    index = np.where((self.mvs[:, 0] < self.block_width * 16) & (self.mvs[:, 1] < self.block_height * 16) &
                                     (self.mvs[:, 1] > self.lim_height) & (self.mvs[:, 3] != 0) &
                                     ((self.mvs[:, 2] > 5) & (self.mvs[:, 0] < self.lim_left_wid[3]) |
                                      (self.mvs[:, 2] < -5) & (self.mvs[:, 0] > self.lim_right_wid[3])))
        filter_mv = self.mvs[index]
        self.remain_red_mvs_index = index[0]
        for mvmv in filter_mv:
            self.mv_x[int(mvmv[1]/16), int(mvmv[0]/16)] = 255

    def _mv_morph_process(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        self.close_x = cv2.morphologyEx(self.mv_x, cv2.MORPH_CLOSE, kernel)
        self.output_x = cv2.connectedComponentsWithStats(self.close_x)

    def _remain_red_area(self):
        for i in range(1, len(self.output_x[2])):
            if self.output_x[2][i][4] >= self.lim_size:
                self.remain_red_mvs

    def _yolo_mv_box_append(self):
        if self.turning == True:
            self.find_direction = True
        if self.find_direction == True and self.turning == False:
            for res in self.output:
                bbox = res['box']
                if bbox[2] < self.lim_left_wid[3] and res['category'] != -1:
                    direct = self._box_direction(bbox)
                    if direct == 1:
                        self.dual_carriageway = False
            if self.dual_carriageway == False:
                self.find_direction = False
            else:
                self.find_direction = True

        arr_x = np.asarray(self.output_x[2])
        keep = np.ones(range(self.output_x[0]).stop, dtype=bool)
        for i in range(self.output_x[0]):
            if self.stopping:
                if not (i > 0 and arr_x[i][4] >= self.lim_size):
                    keep[i] = False
            else:
                if not (i > 0 and arr_x[i][4] >= self.lim_size * 2):
                    keep[i] = False
        arr_x = arr_x[keep]
        for i in range(arr_x.shape[0]):
            call_YOLO = True
            bbox2 = [int(arr_x[i][0] * 16), int(arr_x[i][1] * 16), int(
                (arr_x[i][0] + arr_x[i][2]) * 16), int((arr_x[i][1] + arr_x[i][3]) * 16)]
            for res in self.output:
                bbox = res['box']
                iou = bbox_iou_v2(bbox, bbox2)
                if iou > self.iou_thr:
                    call_YOLO = False
                    break
                # print (iou, call_YOLO)
            if call_YOLO == True:
                mvbox_remove = self._mv_box_remove(bbox2, 0)
                if (mvbox_remove == False):
                    res = {
                        'box': bbox2,
                        'category': -1,
                        'confidence': -1
                    }
                    self.output.append(res)
                    # print("yolo_append")

    def _mv_box_extend(self):
        arr_x = np.asarray(self.output_x[2])
        keep = np.ones(range(self.output_x[0]).stop, dtype=bool)
        for i in range(self.output_x[0]):
            if self.stopping:
                if not (i > 0 and arr_x[i][4] >= self.lim_size):
                    keep[i] = False
            else:
                if not (i > 0 and arr_x[i][4] >= self.lim_size * 2):
                    keep[i] = False
        
        arr_x = arr_x[keep]
        for i in range(arr_x.shape[0]):
            bbox2 = [int(arr_x[i][0] * 16), int(arr_x[i][1] * 16), int(
                (arr_x[i][0] + arr_x[i][2]) * 16), int((arr_x[i][1] + arr_x[i][3]) * 16)]
            # print(self.output)
            if self.output and self.turning == False:
                call_YOLO = True
                tmp_index = -1
                for res in self.output:
                    tmp_index += 1 
                    bbox = res['box']
                    iou = bbox_iou_v2(bbox, bbox2)
                    if iou > self.iou_thr and res['category'] != 0:
                        call_YOLO = False
                        box_extend = False
                        if self.dual_carriageway:
                            # right
                            if bbox2[2] > bbox[2] and bbox[2] > self.lim_right_wid[1]:
                                res2 = {
                                    'box': [bbox[0], bbox[1], bbox2[2], bbox[3]],
                                    'category': res['category'],
                                    'confidence': res['confidence']
                                }
                                box_extend = True
                            # right
                            elif bbox2[0] > bbox[0] and bbox2[2] > bbox[2]:
                                if (bbox2[0] - bbox[2]) > (bbox[2] - bbox[0]) / 3:
                                    res2 = {
                                        'box': [bbox[0], bbox[1], bbox[2] + (bbox[2] - bbox[0]) / 3, bbox[3]],
                                        'category': res['category'],
                                        'confidence': res['confidence']
                                    }
                                    box_extend = True
                                else:
                                    res2 = {
                                        'box': [bbox[0], bbox[1], bbox2[2], bbox[3]],
                                        'category': res['category'],
                                        'confidence': res['confidence']
                                    }
                                    box_extend = True
                            # left
                            elif bbox2[0] < bbox[0] and bbox2[2] < bbox[2]:
                                if (bbox[0] - bbox2[0]) > (bbox[2] - bbox[0]) / 3:
                                    res2 = {
                                        'box': [bbox[0] - ((bbox[2] - bbox[0]) / 3), bbox[1], bbox[2], bbox[3]],
                                        'category': res['category'],
                                        'confidence': res['confidence']
                                    }
                                    box_extend = True
                                else:
                                    res2 = {
                                        'box': [bbox2[0], bbox[1], bbox[2], bbox[3]],
                                        'category': res['category'],
                                        'confidence': res['confidence']
                                    }
                                    box_extend = True
                            # outside
                            elif bbox2[0] < bbox[0] and bbox2[2] > bbox[2]:
                                if (bbox2[2] - bbox2[0]) > (bbox[2] - bbox[0]) * 4 / 3:
                                    if (bbox2[2] - bbox[2]) > (bbox2[0] - bbox[0]):
                                        res2 = {
                                            'box': [bbox2[0], bbox[1], bbox2[0] + ((bbox[2] - bbox[0]) * 4 / 3), bbox[3]],
                                            'category': res['category'],
                                            'confidence': res['confidence']
                                        }
                                        box_extend = True
                                    else:
                                        res2 = {
                                            'box': [bbox2[2] - ((bbox[2] - bbox[0]) * 4 / 3), bbox[1], bbox2[2], bbox[3]],
                                            'category': res['category'],
                                            'confidence': res['confidence']
                                        }
                                        box_extend = True
                                else:
                                    res2 = {
                                        'box': [bbox2[0], bbox[1], bbox2[2], bbox[3]],
                                        'category': res['category'],
                                        'confidence': res['confidence']
                                    }
                                    box_extend = True
                        else:
                            # right
                            if bbox2[2] > bbox[2] and bbox[2] > self.lim_right_wid[3]:
                                res2 = {
                                    'box': [bbox[0], bbox[1], bbox2[2], bbox[3]],
                                    'category': res['category'],
                                    'confidence': res['confidence']
                                }
                                box_extend = True
                            # left
                            elif bbox2[0] < bbox[0] and bbox[0] < self.lim_left_wid[3]:
                                res2 = {
                                    'box': [bbox2[0], bbox[1], bbox[2], bbox[3]],
                                    'category': res['category'],
                                    'confidence': res['confidence']
                                }
                                box_extend = True
                        if box_extend:
                            # self.output.remove(res)
                            # self.output.append(res2)
                            self.output[tmp_index] = res2
                            # print("box_extend")
                        break
                if call_YOLO == True:
                    mvbox_remove = self._mv_box_remove(bbox2, 0)
                    if (mvbox_remove == False):
                        res2 = {
                            'box': bbox2,
                            'category': -1,
                            'confidence': -1
                        }
                        self.output.append(res2)
                        # print("mod")
                        if self.counter > 1:
                            self.counter = -1
        
        #print('self.ori_output_len',self.ori_output_len)
        
        self.retain_index = [(i) for i in range(len(self.output))]
        tmp_index = -1
        for res in self.output:
            tmp_index += 1 
            bbox = res['box']
            box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            for res2 in self.output:
                if (res['category'] < 1 and res2['category'] < 1):
                    bbox2 = res2['box']
                    iou = bbox_iou_v2(bbox, bbox2)
                    if iou > self.iou_thr * 2:
                        box2_area = (bbox2[2] - bbox2[0]) * \
                            (bbox2[3] - bbox2[1])
                        if (box2_area > box_area):
                            self.output.remove(res)
                            self.retain_index.pop(tmp_index)
        for i in range(arr_x.shape[0]):
            bbox2 = [int(arr_x[i][0] * 16), int(arr_x[i][1] * 16), int(
                (arr_x[i][0] + arr_x[i][2]) * 16), int((arr_x[i][1] + arr_x[i][3]) * 16)]
            box2_area = arr_x[i][2] * arr_x[i][3] * 16 * 16
            tmp_index = -1
            for res in self.output:
                tmp_index += 1
                bbox = res['box']
                iou = bbox_iou_v2(bbox, bbox2)
                box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if iou > 0.1 and res['category'] == -1 and box2_area > box_area:
                    res2 = {
                        'box': bbox2,
                        'category': res['category'],
                        'confidence': res['confidence']
                    }
                    # self.output.remove(res)
                    # self.output.append(res2)
                    self.output[tmp_index] = res2
                    # print("box_replace")

    def _mv_box_remove(self, bbox, iou):
        index = np.where((self.mvs[:, 0] > bbox[0]) & (self.mvs[:, 0] < bbox[2]) &
                         (self.mvs[:, 1] > bbox[1]) & (self.mvs[:, 1] < bbox[3]) &
                         (self.mvs[:, 2] != 0))[0]
        #  (self.mvs[:, 2] != 0) & (self.mvs[:, 3] != 0))[0]
        filter_mv = self.mvs[index]
        if len(index) == 0:
            return True
        box_mv_avg_x = sum(filter_mv[:, 2])/len(index)
        box_cen_x = (bbox[0] + bbox[2])/2
        box_cen_gmv_x = self.dir_m_c[1][0] * \
            box_cen_x + self.dir_m_c[1][1]
        # print(box_mv_avg_x, box_cen_gmv_x)
        if len(self.four_direction_lim) == 4:
            if self.turning == True:
                if (box_cen_x < self.lim_left_d[1]):
                    if box_mv_avg_x - box_cen_gmv_x <= 3:
                        return True
                    else:
                        return False
                elif (box_cen_x < self.lim_left_d[3]):
                    box_cen_gmv_x = self.dir_m_c[0][0] * \
                        box_cen_x + self.dir_m_c[0][1]
                    if box_mv_avg_x - box_cen_gmv_x <= 5:
                        return True
                    else:
                        return False
                elif (box_cen_x > self.lim_right_d[3]):
                    box_cen_gmv_x = self.dir_m_c[1][0] * \
                        box_cen_x + self.dir_m_c[1][1]
                    if box_mv_avg_x - box_cen_gmv_x >= -5:
                        return True
                    else:
                        return False
                elif (box_cen_x > self.lim_right_d[1]):
                    box_cen_gmv_x = self.dir_m_c[1][0] * \
                        box_cen_x + self.dir_m_c[1][1]
                    if box_mv_avg_x - box_cen_gmv_x >= -3:
                        return True
                    else:
                        return False
            elif self.dual_carriageway == True:
                if (bbox[0] < 48 or bbox[2] > self.lim_left_d[1] - 48) and bbox[2] < self.half_width:
                    box_cen_y = (bbox[1] + bbox[3])/2
                    if len(index) == 0:
                        return True
                    box_mv_avg_y = sum(filter_mv[:, 3])/len(index)
                    box_cen_gmv_y = self.dir_m_c[0][2] * \
                        box_cen_y + self.dir_m_c[0][3]

                    if box_mv_avg_x - box_cen_gmv_x <= 5 and box_mv_avg_x - box_cen_gmv_x > 0:
                        return True
                    elif box_mv_avg_x - box_cen_gmv_x >= -10 and box_mv_avg_x - box_cen_gmv_x < 0 and \
                            box_mv_avg_y - box_cen_gmv_y <= 5 and box_cen_x < self.lim_left_wid[1]:
                        return True
                    elif box_mv_avg_x - box_cen_gmv_x >= -6 and box_mv_avg_x - box_cen_gmv_x < 0 and \
                            box_mv_avg_y - box_cen_gmv_y <= 5 and box_cen_x < self.lim_left_wid[3]:
                        return True
                    elif box_mv_avg_x - box_cen_gmv_x < -16:
                        return True
                    elif box_mv_avg_x < 0 and iou > 0:
                        return True
                    else:
                        return False
                elif bbox[0] > self.half_width:
                    if box_mv_avg_x >= -5:
                        return True
                    else:
                        return False
                else:
                    return True
            else:
                if bbox[0] < 48 and bbox[2] < self.half_width:
                    box_cen_y = (bbox[1] + bbox[3])/2
                    if len(index) == 0:
                        return True
                    box_mv_avg_y = sum(filter_mv[:, 3])/len(index)
                    box_cen_gmv_y = self.dir_m_c[0][2] * \
                        box_cen_y + self.dir_m_c[0][3]

                    if box_mv_avg_x - box_cen_gmv_x <= 5 and box_mv_avg_x - box_cen_gmv_x > 0:
                        return True
                    elif box_mv_avg_x - box_cen_gmv_x >= -10 and box_mv_avg_x - box_cen_gmv_x < 0 and \
                            box_mv_avg_y - box_cen_gmv_y <= 5 and box_cen_x < self.lim_left_wid[1]:
                        return True
                    elif box_mv_avg_x - box_cen_gmv_x >= -6 and box_mv_avg_x - box_cen_gmv_x < 0 and \
                            box_mv_avg_y - box_cen_gmv_y <= 5 and box_cen_x < self.lim_left_wid[3]:
                        return True
                    elif box_mv_avg_x - box_cen_gmv_x < -16:
                        return True
                    elif box_mv_avg_x < 0 and iou > 0:
                        return True
                    else:
                        return False
                elif bbox[2] < self.half_width:
                    if box_mv_avg_x < 5:
                        return True
                    else:
                        return False
                elif bbox[0] > self.half_width:
                    if box_mv_avg_x > -5:
                        return True
                    else:
                        return False
                else:
                    return True

    def _box_direction(self, bbox):
        index = np.where((self.mvs[:, 0] > bbox[0]) & (self.mvs[:, 0] < bbox[2]) &
                         (self.mvs[:, 1] > bbox[1]) & (self.mvs[:, 1] < bbox[3]) &
                         (self.mvs[:, 2] != 0) & (self.mvs[:, 3] != 0))[0]
        filter_mv = self.mvs[index]
        if (len(index) > 0):
            box_mv_avg_x = sum(filter_mv[:, 2])/len(index)
            box_cen_x = (bbox[0] + bbox[2])/2
            box_cen_gmv_x = self.dir_m_c[0][0] * \
                box_cen_x + self.dir_m_c[0][1]
            if (box_mv_avg_x - box_cen_gmv_x) > 0:
                return 1
            elif (box_mv_avg_x - box_cen_gmv_x) < 0:
                return -1
            else:
                return 0
        else:
            return -1

    def _sub_mv_mod(self, boundingbox_list, json_path, counter, img_width, img_height, dir_m_c=None, y_lim=None, draw_mv_in_bbox=False):
        if boundingbox_list == None:
            if self.draw_mv_in_bbox:
                return None, 0, self.residual, self.turning, self.stopping, self.dual_carriageway, self.dir_m_c, \
                    self.gmv, self.lim_left_d[3], self.lim_right_d[3], self.close_x
            else:
                return None, 0, self.residual, self.turning, self.stopping, self.dual_carriageway, self.dir_m_c
        self.bbox_list = [det['box'][:4] if np.isnan(det['box']).any() != True else [
        ] for det in boundingbox_list]
        self.output = boundingbox_list
        self.draw_mv_in_bbox = draw_mv_in_bbox
        self.img_width = img_width
        self.img_height = img_height
        self.half_width = int(img_width / 2)
        self.half_height = int(img_height / 2)
        self.block_width = int(img_width / 16)
        self.block_height = int(img_height / 16)
        self.lim_left_wid = [self.img_width/10, self.img_width/5,
                             self.img_width*3/10, self.img_width*2/5]
        self.lim_right_wid = [self.img_width*9/10, self.img_width*4/5,
                              self.img_width*7/10, self.img_width*3/5]
        if y_lim:
            self.lim_height = int(y_lim / 2)
        else:
            self.lim_height = int(img_height / 2)
        self.lim_size = 25
        self.iou_thr = 0.3
        self.counter = counter
        # self.half_searchwindow = 8
        self.gmv_index = []
        self.dir_m_c = dir_m_c
        self.residual = []

        if not self.dir_m_c:
            self.x_lim = None
            self.y_lim = y_lim

        self.direction_length = 4  # 2
        self.direction_area = [2, 3, 0, 1]  # [0]
        if self.bbox_list:  # bbox_list is not empty
            self._json_load_to_array(json_path)
            start_t = time.time()
            self._remove_bbox_mv()
            if not self.dir_m_c:
                self._global_mv_four_quadrant()
            else:
                self._cal_residual()

        # t = time.time()
        self._mv_map_generate()
        self._mv_morph_process()
        self._yolo_mv_box_append()
        # print ("all_mod_mv: %.3f ms" % ((time.time()-t) * 1000))

        mv_t = time.time()-start_t
        if self.draw_mv_in_bbox:
            if self.output:
                for res in self.output:
                    bbox = res['box']
                    if bbox[2] < self.lim_left_wid[3]:
                        dirct = self._box_direction(bbox)
                        res['box'].append(dirct)
                    else:
                        res['box'].append(0)
            self.gmv = self._draw_mv(
                self.mv_src_copy[self.gmv_index, :], self.mv_d_copy[self.gmv_index, :])
            return self.output, mv_t, self.residual, self.turning, self.stopping, self.dual_carriageway, self.dir_m_c, \
                self.gmv, self.lim_left_d[3], self.lim_right_d[3], self.close_x
        else:
            return self.output, mv_t, self.residual, self.turning, self.stopping, self.dual_carriageway, self.dir_m_c

    def __call__(self, boundingbox_list, json_path, counter, dual_carriageway, img_width, img_height, dir_m_c=None, y_lim=None, integer=False, mod_enable=True, draw_mv_in_bbox=False):
        if boundingbox_list == None:
            if mod_enable:
                if self.draw_mv_in_bbox:
                    return None, self.counter, 0, self.residual, self.turning, self.stopping, self.dir_m_c, self.draw_mv, self.gmv, \
                        self.lim_left_d[3], self.lim_right_d[3], self.close_x, self.disappear_index, None
                else:
                    return None, self.counter, 0, self.residual, self.turning, self.stopping, self.dir_m_c, self.disappear_index, None
            else:
                if self.draw_mv_in_bbox:
                    return None, self.counter, 0, self.residual, self.dir_m_c, self.draw_mv, self.gmv, self.disappear_index, None
                else:
                    return None, self.counter, 0, self.residual, self.dir_m_c, self.disappear_index, None
        tt = time.time()
        self.bbox_list = [det['box'][:4] if np.isnan(det['box']).any() != True else [
        ] for det in boundingbox_list]
        self.output = deepcopy(boundingbox_list)
        self.interger = integer
        self.draw_mv_in_bbox = draw_mv_in_bbox
        self.img_width = img_width
        self.img_height = img_height
        self.half_width = int(img_width / 2)
        self.half_height = int(img_height / 2)
        self.block_width = int(img_width / 16)
        self.block_height = int(img_height / 16)
        self.lim_left_wid = [self.img_width/10, self.img_width/5,
                             self.img_width*3/10, self.img_width*2/5]
        self.lim_right_wid = [self.img_width*9/10, self.img_width*4/5,
                              self.img_width*7/10, self.img_width*3/5]
        if y_lim:
            self.lim_height = int(y_lim / 2)
        else:
            self.lim_height = int(img_height / 2)
        self.lim_size = 25
        self.iou_thr = 0.3
        self.counter = counter
        self.half_searchwindow = 0
        self.gmv_index = []
        self.dir_m_c = dir_m_c
        self.residual = []
        self.mv_class = []
        self.mv_class_final = []

        if not self.dir_m_c:
            self.x_lim = None
            self.y_lim = y_lim

        self.direction_length = 4  # 2
        self.direction_area = [2, 3, 0, 1]  # [0]
        self.dual_carriageway = dual_carriageway
        self.disappear_index = []
        self.find_direction = True
        # left up, reight up, left down, right down [xmin,xmax,ymin,ymax]
        if self.bbox_list:  # bbox_list is not empty
            self._json_load_to_array(json_path)
            start_t = time.time()
            self._remove_bbox_mv()
            if not self.dir_m_c:
                self._global_mv_four_quadrant()
            else:
                self._cal_residual()
            self._mv_map_generate()
            new_bbox = self._get_bbox_mv()
            if self.interger == True:
                new_bbox = self._box_to_int(new_bbox)
            i = 0
            j = 0
            for box in new_bbox:
                self.output[i]['box'] = box
                if box == []:
                    self.disappear_index.append(i)
                    self.mv_class_final.append(-1)
                else:
                    self.mv_class_final.append(self.mv_class[j])
                    j += 1
                i += 1
        self.output = [d for d in self.output if d['box'] != []]
        self.output = None if (list(filter(None, self.output)) == []) else list(
            filter(None, self.output))
        
        self.mv_class_final = np.array(self.mv_class_final)
        self.mv_class = np.array(self.mv_class_final[self.mv_class_final >= 0])
        self.mv_class = None if (
            list(self.mv_class) == []) else list(self.mv_class)
        self.retain_index = []

        if mod_enable:
            # t = time.time()
            # self._mv_map_generate()
            self._mv_morph_process()
            if self.output:
                self._mv_box_extend()
            # self._mv_box_append()
            # print ("all_mod_mv: %.3f ms" % ((time.time()-t) * 1000))

        mv_t = time.time()-start_t
        # plt.clf()
        if mod_enable:
            self.ori_output_len = [(i) for i in range(len(self.output))]
            self.disappear_index += list(set(self.ori_output_len)-set(self.retain_index))
            if self.draw_mv_in_bbox:
                if self.output:
                    for res in self.output:
                        bbox = res['box']
                        if bbox[2] < self.lim_left_wid[3]:
                            dirct = self._box_direction(bbox)
                            res['box'].append(dirct)
                        else:
                            res['box'].append(0)
                self.gmv = self._draw_mv(
                    self.mv_src_copy[self.gmv_index, :], self.mv_d_copy[self.gmv_index, :])
                
                return self.output, self.counter, mv_t, self.residual, self.turning, self.stopping, self.dir_m_c, self.draw_mv, self.gmv, \
                    self.lim_left_d[3], self.lim_right_d[3], self.close_x, self.disappear_index, self.mv_class
            else:
                return self.output, self.counter, mv_t, self.residual, self.turning, self.stopping, self.dir_m_c, \
                    self.disappear_index, self.mv_class
        else:
            if self.draw_mv_in_bbox:
                self.gmv = self._draw_mv(
                    self.mv_src_copy[self.gmv_index, :], self.mv_d_copy[self.gmv_index, :])
                return self.output, self.counter, mv_t, self.residual, self.dir_m_c, self.draw_mv, self.gmv, \
                    self.disappear_index, self.mv_class
            else:
                return self.output, self.counter, mv_t, self.residual, self.dir_m_c, \
                    self.disappear_index, self.mv_class
