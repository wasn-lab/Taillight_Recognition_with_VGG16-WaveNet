/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// headers in STL
#include <chrono>
#include <cmath>
#include <iostream>

// headers in PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_ros/transforms.h>

// headers in ROS
#include <tf/transform_datatypes.h>

// headers in local files
#include <std_msgs/Float64.h>
#include "msgs/DetectedObjectArray.h"
#include "detected_object_class_id.h"
#include "fusion_source_id.h"
#include "lidar_point_pillars/ped_cyc/point_pillars_ros_ped_cyc.h"

// for StopWatch
#include <iostream>
#include <pcl/common/time.h>

// for obbs
#include <visualization_msgs/Marker.h>

pcl::StopWatch g_stopWatch_Ped_Cyc;

PointPillarsROS_Ped_Cyc::PointPillarsROS_Ped_Cyc()
  : private_nh_("~")
  , has_subscribed_baselink_(false)
  , NUM_POINT_FEATURE_(4)
  , OUTPUT_NUM_BOX_FEATURE_(7)
  , TRAINED_SENSOR_HEIGHT_(1.73f)
  , NORMALIZING_INTENSITY_VALUE_(1.0f)
  , BASELINK_FRAME_("base_link")
{
  //ros related param
  private_nh_.param<bool>("baselink_support", baselink_support_, true);

  //algorithm related params
  private_nh_.param<bool>("reproduce_result_mode", reproduce_result_mode_, false);
  private_nh_.param<float>("score_threshold", score_threshold_, 0.5f);
  private_nh_.param<float>("nms_overlap_threshold", nms_overlap_threshold_, 0.5f);
  private_nh_.param<std::string>("pfe_onnx_file", pfe_onnx_file_, "");
  private_nh_.param<std::string>("rpn_onnx_file", rpn_onnx_file_, "");

  point_pillars_ptr_.reset(new PointPillars_Ped_Cyc(reproduce_result_mode_, score_threshold_, nms_overlap_threshold_,
                                            pfe_onnx_file_, rpn_onnx_file_));
}

void PointPillarsROS_Ped_Cyc::createROSPubSub()
{
  sub_points_ = nh_.subscribe<sensor_msgs::PointCloud2>("/points_raw", 1, &PointPillarsROS_Ped_Cyc::pointsCallback, this);
  pub_objects_ = nh_.advertise<msgs::DetectedObjectArray>("/LidarDetection/Ped_Cyc", 1);
}

geometry_msgs::Pose PointPillarsROS_Ped_Cyc::getTransformedPose(const geometry_msgs::Pose& in_pose, const tf::Transform& tf)
{
  tf::Transform transform;
  geometry_msgs::PoseStamped out_pose;
  transform.setOrigin(tf::Vector3(in_pose.position.x, in_pose.position.y, in_pose.position.z));
  transform.setRotation(
  tf::Quaternion(in_pose.orientation.x, in_pose.orientation.y, in_pose.orientation.z, in_pose.orientation.w));
  geometry_msgs::PoseStamped pose_out;
  tf::poseTFToMsg(tf * transform, out_pose.pose);
  return out_pose.pose;
}

void PointPillarsROS_Ped_Cyc::pubDetectedObject(const std::vector<float>& detections, const std::vector<float>& scores, const std::vector<int>& labels, const std_msgs::Header& in_header)
{
  msgs::DetectedObjectArray MsgObjArr;
  MsgObjArr.header = in_header;
  int num_objects = detections.size() / OUTPUT_NUM_BOX_FEATURE_;
  if (num_objects > 0)
  {
    for (size_t i = 0; i < num_objects; i++)
    {
      msgs::DetectedObject object;

      object.score = scores[i];

      float center_x = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 0];
      float center_y = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 1];
      float center_z = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 2];
      
      float dimension_x = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 5]; //4
      float dimension_y = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 3]; //3
      float dimension_z = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 4]; //5

      object.center_point.x = center_x;
      object.center_point.y = center_y;
      object.center_point.z = center_z;

      // heading
      float yaw = detections[i * OUTPUT_NUM_BOX_FEATURE_ + 6];
      yaw += M_PI/2;
      yaw = std::atan2(std::sin(yaw), std::cos(yaw));
      geometry_msgs::Quaternion q = tf::createQuaternionMsgFromYaw(-yaw);
      object.heading.x = q.x;
      object.heading.y = q.y;
      object.heading.z = q.z;
      object.heading.w = q.w;

      // dimension
      object.dimension.length = dimension_x;
      object.dimension.width = dimension_y;
      object.dimension.height = dimension_z;

      // bPoint
      visualization_msgs::Marker box;
      std::vector<Eigen::Vector3f> p;
      p.resize(8);
      float x = dimension_x/2.0f;
      float y = dimension_y/2.0f;
      float z = dimension_z/2.0f;
      p[0] = Eigen::Vector3f(-x, -y, -z);
      p[1] = Eigen::Vector3f(-x, -y, z);
      p[2] = Eigen::Vector3f( -x, y, z);
      p[3] = Eigen::Vector3f( -x, y, -z);
      p[4] = Eigen::Vector3f( x, -y, -z);
      p[5] = Eigen::Vector3f( x, -y, z);
      p[6] = Eigen::Vector3f( x, y, z);
      p[7] = Eigen::Vector3f(x, y, -z);
      Eigen::Vector3f center(center_x, center_y, center_z);
      Eigen::Quaternionf quat(q.w, q.x, q.y, q.z);
      for (int i = 0; i < 8; i++)
      {
        p[i] = quat * p[i];
        p[i] += center;
      }

      object.bPoint.p0.x = p[0](0);
      object.bPoint.p0.y = p[0](1);
      object.bPoint.p0.z = p[0](2);

      object.bPoint.p1.x = p[1](0);
      object.bPoint.p1.y = p[1](1);
      object.bPoint.p1.z = p[1](2);

      object.bPoint.p2.x = p[2](0);
      object.bPoint.p2.y = p[2](1);
      object.bPoint.p2.z = p[2](2);

      object.bPoint.p3.x = p[3](0);
      object.bPoint.p3.y = p[3](1);
      object.bPoint.p3.z = p[3](2);

      object.bPoint.p4.x = p[4](0);
      object.bPoint.p4.y = p[4](1);
      object.bPoint.p4.z = p[4](2);

      object.bPoint.p5.x = p[5](0);
      object.bPoint.p5.y = p[5](1);
      object.bPoint.p5.z = p[5](2);

      object.bPoint.p6.x = p[6](0);
      object.bPoint.p6.y = p[6](1);
      object.bPoint.p6.z = p[6](2);

      object.bPoint.p7.x = p[7](0);
      object.bPoint.p7.y = p[7](1);
      object.bPoint.p7.z = p[7](2);

      if (labels[i] == 1)
      {
        //object.label = "Cyclist";
        object.classId = sensor_msgs_itri::DetectedObjectClassId::Motobike;
      }
      else if (labels[i] == 2)
      {
        //object.label = "Pedestrian";
        object.classId = sensor_msgs_itri::DetectedObjectClassId::Person;
      }
      else
      {
        //object.label = "Car";
        object.classId = sensor_msgs_itri::DetectedObjectClassId::Car;
      }
      
      // if (baselink_support_)
      // {
      //   object.pose = getTransformedPose(object.pose, angle_transform_inversed_);
      // }

      // base info
      object.header = in_header;
      object.fusionSourceId = sensor_msgs_itri::FusionSourceId::Lidar;

      // pub
      MsgObjArr.objects.push_back(object);
    }
    pub_objects_.publish(MsgObjArr);
    std::cout << "[PPilars_PedCyc]: " << g_stopWatch_Ped_Cyc.getTimeSeconds() << 's' << std::endl;
  }
}

void PointPillarsROS_Ped_Cyc::getBaselinkToLidarTF(const std::string& target_frameid)
{
  try
  {
    tf_listener_.waitForTransform(BASELINK_FRAME_, target_frameid, ros::Time(0), ros::Duration(1.0));
    tf_listener_.lookupTransform(BASELINK_FRAME_, target_frameid, ros::Time(0), baselink2lidar_);
    analyzeTFInfo(baselink2lidar_);
    has_subscribed_baselink_ = true;
  }
  catch (tf::TransformException ex)
  {
    ROS_ERROR("%s", ex.what());
  }
}

void PointPillarsROS_Ped_Cyc::analyzeTFInfo(tf::StampedTransform baselink2lidar)
{
  tf::Vector3 v = baselink2lidar.getOrigin();
  offset_z_from_trained_data_ = v.getZ() - TRAINED_SENSOR_HEIGHT_;

  tf::Quaternion q = baselink2lidar_.getRotation();
  angle_transform_ = tf::Transform(q);
  angle_transform_inversed_ = angle_transform_.inverse();
}

void PointPillarsROS_Ped_Cyc::pclToArray(const pcl::PointCloud<pcl::PointXYZI>::Ptr& in_pcl_pc_ptr, float* out_points_array,
                                 const float offset_z)
{
  for (size_t i = 0; i < in_pcl_pc_ptr->size(); i++)
  {
    pcl::PointXYZI point = in_pcl_pc_ptr->at(i);
    out_points_array[i * NUM_POINT_FEATURE_ + 0] = point.x;
    out_points_array[i * NUM_POINT_FEATURE_ + 1] = point.y;
    out_points_array[i * NUM_POINT_FEATURE_ + 2] = point.z + offset_z;
    out_points_array[i * NUM_POINT_FEATURE_ + 3] = float(point.intensity / NORMALIZING_INTENSITY_VALUE_);
  }
}

void PointPillarsROS_Ped_Cyc::pointsCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  // check data from hardware
  // if ((ros::Time::now().toSec() - msg->header.stamp.toSec()) < 3600)
  // {
    // uint64_t diff_time = (ros::Time::now().toSec() - msg->header.stamp.toSec()) * 1000;
    // std::cout << "[Top->Pillars_ped]: " << diff_time << "ms" << std::endl;
  // }
  g_stopWatch_Ped_Cyc.reset();

  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*msg, *pcl_pc_ptr);

  if (baselink_support_)
  {
    if (!has_subscribed_baselink_)
    {
      getBaselinkToLidarTF(msg->header.frame_id);
    }
    pcl_ros::transformPointCloud(*pcl_pc_ptr, *pcl_pc_ptr, angle_transform_);
  }

  float* points_array = new float[pcl_pc_ptr->size() * NUM_POINT_FEATURE_];
  if (baselink_support_ && has_subscribed_baselink_)
  {
    pclToArray(pcl_pc_ptr, points_array, offset_z_from_trained_data_);
  }
  else
  {
    pclToArray(pcl_pc_ptr, points_array);
  }

  std::vector<float> out_detection;
  std::vector<float> out_score;
  std::vector<int> out_label;
  point_pillars_ptr_->doInference(points_array, pcl_pc_ptr->size(), out_detection, out_score, out_label);

  delete[] points_array;
  pubDetectedObject(out_detection, out_score, out_label, msg->header);
}
