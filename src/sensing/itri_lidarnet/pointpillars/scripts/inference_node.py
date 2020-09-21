import rospy
from msgs.msg import (DetectedObject, DetectedObjectArray, PointXYZ)
from visualization_msgs.msg import (MarkerArray, Marker)
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

import io as sysio
import json
import os
import pickle
import sys
import time
from functools import partial
from pathlib import Path
import fire
import matplotlib.pyplot as plt
import numba
import numpy as np

import second.core.box_np_ops as box_np_ops
import second.core.preprocess as prep
from second.core.anchor_generator import AnchorGeneratorStride
from second.core.box_coders import GroundBox3dCoder
from second.core.point_cloud.point_cloud_ops import points_to_voxel
from second.core.region_similarity import (
    DistanceSimilarity, NearestIouSimilarity, RotateIouSimilarity)
from second.core.sample_ops import DataBaseSamplerV2
from second.core.target_assigner import TargetAssigner
from second.data import kitti_common as kitti
from second.kittiviewer.glwidget import KittiGLViewWidget
from second.protos import pipeline_pb2
from second.utils import bbox_plot
from second.utils.bbox_plot import GLColor
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.pytorch.inference import TorchInferenceContext
from second.utils.progress_bar import list_bar

result_pub = rospy.Publisher ('/LidarDetection_PointPillars', DetectedObjectArray, queue_size=30)
marker_pub = rospy.Publisher('/LidarDetection_PointPillars/bbox', MarkerArray, queue_size=10)

def corners_lidar(dims, origin=[0.5, 0.5, 0.0]):
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1).astype(
            dims.dtype)
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # (converted 3d) x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # translate according to origin
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners

def rotation_3d_in_axis(points, angles, axis=2):
    # points: [N, 8, 3]
    rot_sin = np.sin(np.pi/2-angles)
    rot_cos = np.cos(np.pi/2-angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")
    
    # result: [N, 8, 3]
    return np.einsum('aij,jka->aik', points, rot_mat_T)

def center_to_corner_lidar(centers,
                            dims,
                            angles=None,
                            origin=[0.5, 0.5, 0],
                            axis=2):
    corners = corners_lidar(dims, origin=origin)
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    # corners: [N, 8, 3]
    return corners

class Inference_node:
    def __init__(self):
        self.inference_ctx = None
        self.points = None
    
    def build(self, config_path_str):
        self.inference_ctx = TorchInferenceContext()
        vconfig_path = Path(config_path_str)
        self.inference_ctx.build(vconfig_path)
        rospy.loginfo("[pointpillars] Build PointPillars succeed.")
    
    def  loadckpt(self, ckpt_path_str):
        ckpt_path = Path(ckpt_path_str)
        self.inference_ctx.restore(ckpt_path)
        rospy.loginfo("[pointpillars] load PointPillars ckpt succeed.")

    def callback_lidarAll(self, msg):
        # This is where pp inference code put in 
        
        # first read point cloud array from message
        #dtype_list = [(f.name, np.float32) for f in msg.fields]
        dtype_list = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('blank', np.float32), ('intensity', np.float32),
                                    ('blank2', np.float32), ('blank3', np.float32), ('blank4', np.float32)]
        points = np.fromstring(msg.data, dtype_list)
        points = np.array(points.tolist())[:, [0, 1, 2, 4]]
        # intensity_max = np.max(points[:,3])
        # print ("max: ", intensity_max)
        # points[:,3] = points[:,3] * 0
        points[:,3] = np.maximum(points[:,3]-1000, 0)  * 255/1500
        np.reshape(points, (msg.height, msg.width, -1))
        lidar_time = msg.header.stamp.secs
        #stamp = msg.header.stamp
        # check input feature size and feature name
        
        assert dtype_list[4][0] == "intensity" or dtype_list[4][0] == "reflection"
        # convert input into preprocessed example
        t = time.time()
        inputs = self.inference_ctx.get_inference_input_dict_without_kitti(lidar_time, points)
        rospy.loginfo("[pointpillars] input preparation time: %fs", time.time()-t)
        t = time.time()
        with self.inference_ctx.ctx():
            det_annos = self.inference_ctx.inference_lidar(inputs)
        rospy.loginfo("[pointpillars] detection time: %fs", time.time()-t)
        num_anno = det_annos[0]["name"].shape[0]
        anno = det_annos[0]
        #print(det_annos)

        det_object_array = DetectedObjectArray()
        det_marker_array = MarkerArray()
        id = 0

        for i in range(num_anno):
            det_object = DetectedObject()
            det_object.header = msg.header
            classID = anno["name"][i]
            if classID == "Car":
                det_object.classId = 4
            elif classID == "Pedestrian":
                det_object.classId = 1
            elif classID == "Cyclist":
                det_object.classId = 3
            # center is in [N, 3]
            bCenter = np.array(anno["location"][i]).reshape([-1, 3])
            # bCenter[:, 0] = -bCenter[:,0]
            # print(bCenter)
            bDims = np.array([anno["dimensions"][i]]).reshape([-1, 3])
            bRotation = np.array([anno["rotation_y"][i]]).reshape([-1])
            #bCorners: [N, 8, 3]
            bCorners = center_to_corner_lidar(bCenter, bDims[:,[1, 0, 2]], bRotation, axis=2)
            det_object.lidarInfo.boxCenter.x = bCenter[0][0]
            det_object.lidarInfo.boxCenter.y = bCenter[0][1]
            det_object.lidarInfo.boxCenter.z = bCenter[0][2]
            # print(bDims.shape)
            # det_object.lidarInfo.width = bDims[0][1]
            # det_object.lidarInfo.height = bDims[0][2]
            det_object.bPoint.p0 = PointXYZ(bCorners[0][0][0], bCorners[0][0][1], bCorners[0][0][2])
            det_object.bPoint.p1 = PointXYZ(bCorners[0][1][0], bCorners[0][1][1], bCorners[0][1][2])
            det_object.bPoint.p2 = PointXYZ(bCorners[0][2][0], bCorners[0][2][1], bCorners[0][2][2])
            det_object.bPoint.p3 = PointXYZ(bCorners[0][3][0], bCorners[0][3][1], bCorners[0][3][2])
            det_object.bPoint.p4 = PointXYZ(bCorners[0][4][0], bCorners[0][4][1], bCorners[0][4][2])
            det_object.bPoint.p5 = PointXYZ(bCorners[0][5][0], bCorners[0][5][1], bCorners[0][5][2])
            det_object.bPoint.p6 = PointXYZ(bCorners[0][6][0], bCorners[0][6][1], bCorners[0][6][2])
            det_object.bPoint.p7 = PointXYZ(bCorners[0][7][0], bCorners[0][7][1], bCorners[0][7][2])
            rot_sin = np.sin(bRotation[0]/2)
            rot_cos = np.cos(bRotation[0]/2)
            det_object.bOrient.w,det_object.bOrient.x, det_object.bOrient.y, det_object.bOrient.z = rot_cos, 0.0, 0.0, rot_sin
            det_object.header = msg.header
            det_object.header.frame_id = "lidar"
            det_object_array.objects.append(det_object)

            if (np.all(bCenter[0] > np.array([-69.12, -39.68, -4]))) and (np.all(bCenter[0] <np.array([69.12, 39.68, 0]))) and (np.all(bDims[0] < np.array([10.0, 10.0, 10.0]))) and (anno["score"][i] > 0.0):
                det_marker = Marker()
                det_marker.type = Marker.LINE_LIST
                det_marker.header = msg.header
                det_marker.header.frame_id = "lidar"
                det_marker.id = id
                det_marker.lifetime = rospy.Duration(0.2)
                # det_marker.scale.x, det_marker.scale.y, det_marker.scale.z = bDims[0][0], bDims[0][1], bDims[0][2]
                det_marker.scale.x = 0.1
                # det_marker.pose.position.x, det_marker.pose.position.y, det_marker.pose.position.z = bCenter[0][0], bCenter[0][1], bCenter[0][2]
                det_marker.pose.orientation.w, det_marker.pose.orientation.x, det_marker.pose.orientation.y, det_marker.pose.orientation.z = 1.0, 0.0, 0.0, 0.0
                for i in range(4):
                    p, p_next = Point(), Point()
                    p.x, p.y, p.z = bCorners[0][i][0], bCorners[0][i][1], bCorners[0][i][2]
                    if i != 3:
                        p_next.x, p_next.y, p_next.z = bCorners[0][i+1][0], bCorners[0][i+1][1], bCorners[0][i+1][2]
                    else:
                        p_next.x, p_next.y, p_next.z = bCorners[0][0][0], bCorners[0][0][1], bCorners[0][0][2]
                    det_marker.points.append(p)
                    det_marker.points.append(p_next)
                for i in range(4, 8):
                    p, p_next = Point(), Point()
                    p.x, p.y, p.z = bCorners[0][i][0], bCorners[0][i][1], bCorners[0][i][2]
                    if i != 7:
                        p_next.x, p_next.y, p_next.z = bCorners[0][i+1][0], bCorners[0][i+1][1], bCorners[0][i+1][2]
                    else:
                        p_next.x, p_next.y, p_next.z = bCorners[0][4][0], bCorners[0][4][1], bCorners[0][4][2]
                    det_marker.points.append(p)
                    det_marker.points.append(p_next)
                for i in range(4):
                    p, p_next = Point(), Point()
                    p.x, p.y, p.z = bCorners[0][i][0], bCorners[0][i][1], bCorners[0][i][2]
                    p_next.x, p_next.y, p_next.z = bCorners[0][i+4][0], bCorners[0][i+4][1], bCorners[0][i+4][2]
                    det_marker.points.append(p)
                    det_marker.points.append(p_next)
                if det_object.classId == 4:
                    det_marker.color.a, det_marker.color.r, det_marker.color.g, det_marker.color.b = 1.0, 0.0, 1.0, 0.0
                elif det_object.classId == 1:
                    det_marker.color.a, det_marker.color.r, det_marker.color.g, det_marker.color.b = 1.0, 0.0, 1.0, 1.0
                elif det_object.classId == 3:
                    det_marker.color.a, det_marker.color.r, det_marker.color.g, det_marker.color.b = 1.0, 1.0, 0.0, 1.0
                det_marker_array.markers.append(det_marker)
                id += 1

        result_pub.publish(det_object_array)
        if det_marker_array.markers is not None:
            marker_pub.publish(det_marker_array)

def main():
    rospy.init_node('pointpillars_inference')
    rospy.loginfo("[pointpillars] inference node running")
    
    # Inference node object
    infer_node = Inference_node()
    # infer_node.build("./second/configs/pointpillars/car/xyres_16.proto")
    # infer_node.loadckpt("./second/model_20200722/eval_checkpoints/voxelnet-296960.tckpt")
    infer_node.build("./second/configs/pointpillars/ped_cycle/xyres_16.proto")
    infer_node.loadckpt("./second/model_ped_cycle_20200803/eval_checkpoints/voxelnet-294556.tckpt")
    
    rospy.Subscriber("/LidarAll/NonGround", PointCloud2, infer_node.callback_lidarAll)

    rospy.spin()
        

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass