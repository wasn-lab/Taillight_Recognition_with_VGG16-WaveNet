#include "to_tf_map.h"

namespace to_tf_map
{
void ToTFMap::convert(msgs::PointXYZ& p, const geometry_msgs::TransformStamped tf_stamped)
{
  // TF (lidar-to-map) for object pose
  geometry_msgs::Pose pose_in_lidar;
  pose_in_lidar.position.x = p.x;
  pose_in_lidar.position.y = p.y;
  pose_in_lidar.position.z = p.z;
  pose_in_lidar.orientation.x = 0;
  pose_in_lidar.orientation.y = 0;
  pose_in_lidar.orientation.z = 0;
  pose_in_lidar.orientation.w = 1;

  geometry_msgs::Pose pose_in_map;
  tf2::doTransform(pose_in_lidar, pose_in_map, tf_stamped);
  p.x = pose_in_map.position.x;
  p.y = pose_in_map.position.y;
  p.z = pose_in_map.position.z;
}

void ToTFMap::convert_all_to_map_tf(std::vector<msgs::DetectedObject>& objs)
{
  for (auto& obj : objs)
  {
    if (obj.header.frame_id != frame_id_target_)
    {
      geometry_msgs::TransformStamped tf_stamped;

      try
      {
        tf_stamped = tf_buffer_.lookupTransform(frame_id_target_, frame_id_source_, obj.header.stamp);
      }
      catch (tf2::TransformException& ex1)
      {
        ROS_WARN("%s", ex1.what());
        try
        {
          tf_stamped = tf_buffer_.lookupTransform(frame_id_target_, frame_id_source_, ros::Time(0));
        }
        catch (tf2::TransformException& ex2)
        {
          ROS_WARN("%s", ex2.what());
          return;
        }
      }

      // convert bPoint
      convert(obj.bPoint.p0, tf_stamped);
      convert(obj.bPoint.p1, tf_stamped);
      convert(obj.bPoint.p2, tf_stamped);
      convert(obj.bPoint.p3, tf_stamped);
      convert(obj.bPoint.p4, tf_stamped);
      convert(obj.bPoint.p5, tf_stamped);
      convert(obj.bPoint.p6, tf_stamped);
      convert(obj.bPoint.p7, tf_stamped);

      // convert center_point
      obj.center_point.x = (obj.bPoint.p0.x + obj.bPoint.p6.x) / 2;
      obj.center_point.y = (obj.bPoint.p0.y + obj.bPoint.p6.y) / 2;
      obj.center_point.z = (obj.bPoint.p0.z + obj.bPoint.p6.z) / 2;

      // convert cPoint
      if (!obj.cPoint.lowerAreaPoints.empty())
      {
        for (auto& p : obj.cPoint.lowerAreaPoints)
        {
          convert(p, tf_stamped);
        }
      }

      for (auto& state : obj.track.states)
      {
        state.header.frame_id = frame_id_target_;
      }

      // convert PP
      if (obj.track.is_ready_prediction)
      {
        for (int i = 0; i < num_forecasts_; i++)
        {
          msgs::PointXYZ p;
          p.x = obj.track.forecasts[i].position.x;
          p.y = obj.track.forecasts[i].position.y;
          p.z = 0.;

          convert(p, tf_stamped);

          obj.track.forecasts[i].position.x = p.x;
          obj.track.forecasts[i].position.y = p.y;
        }
      }

      obj.header.frame_id = frame_id_target_;
    }
  }
}

void ToTFMap::callbackDetection(const msgs::DetectedObjectArray::ConstPtr& input)
{
  msgs::DetectedObjectArray output;

  output.header = input->header;

  // set output.header.frame_id and frame_id_source_
  frame_id_source_ = "base_link";
  if (output.header.frame_id != "lidar" && output.header.frame_id != "base_link")
  {
    frame_id_source_ = output.header.frame_id;
  }
  output.header.frame_id = frame_id_target_;

  output.objects.assign(input->objects.begin(), input->objects.end());
  convert_all_to_map_tf(output.objects);

  std::cout << "AAA" << std::endl;
  pub_to_tf_map_.publish(output);
}

int ToTFMap::run()
{
  std::string domain = "/itri_to_tf_map/";
  nh_.param<std::string>(domain + "in_topic", in_topic_, "Tracking3D");
  nh_.param<int>(domain + "num_forecasts", num_forecasts_, 20);

  sub_detection_ = nh_.subscribe(in_topic_, 1, &ToTFMap::callbackDetection, this);
  pub_to_tf_map_ = nh_.advertise<msgs::DetectedObjectArray>(in_topic_ + "/tf_map", 1);

  tf2_ros::TransformListener tf_listener(tf_buffer_);

  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
}  // namespace to_tf_map
