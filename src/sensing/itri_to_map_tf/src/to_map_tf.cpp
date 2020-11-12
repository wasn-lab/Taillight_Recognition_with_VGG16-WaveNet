#include "to_map_tf.h"

namespace to_map_tf
{
void ToMapTF::convert(msgs::PointXYZ& p, const geometry_msgs::TransformStamped tf_stamped)
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

void ToMapTF::callbackDetection(const msgs::DetectedObjectArray::ConstPtr& input)
{
  msgs::DetectedObjectArray output;
  output.header = input->header;
  output.header.frame_id = frame_id_target_;

  output.objects.assign(input->objects.begin(), input->objects.end());

  for (auto& obj : output.objects)
  {
    if (obj.header.frame_id != frame_id_target_)
    {
      geometry_msgs::TransformStamped tf_stamped;
      frame_id_source_ = obj.header.frame_id;

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

      convert(obj.bPoint.p0, tf_stamped);
      convert(obj.bPoint.p1, tf_stamped);
      convert(obj.bPoint.p2, tf_stamped);
      convert(obj.bPoint.p3, tf_stamped);
      convert(obj.bPoint.p4, tf_stamped);
      convert(obj.bPoint.p5, tf_stamped);
      convert(obj.bPoint.p6, tf_stamped);
      convert(obj.bPoint.p7, tf_stamped);

      if (!obj.cPoint.lowerAreaPoints.empty())
      {
        for (auto p : obj.cPoint.lowerAreaPoints)
        {
          convert(p, tf_stamped);
        }
      }

      obj.header.frame_id = frame_id_target_;
    }
  }

  pub_to_map_tf_.publish(output);
}

int ToMapTF::run()
{
  std::string topic = "CameraDetection";

  sub_detection_ = nh_.subscribe("topic", 1, &ToMapTF::callbackDetection, this);
  pub_to_map_tf_ = nh_.advertise<msgs::DetectedObjectArray>(topic + std::string("/to_map_tf"), 1);

  tf2_ros::TransformListener tf_listener(tf_buffer_);

  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
}  // namespace to_map_tf
