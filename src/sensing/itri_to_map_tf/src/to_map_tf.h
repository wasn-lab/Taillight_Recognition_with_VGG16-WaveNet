#ifndef __TO_MAP_TF_H__
#define __TO_MAP_TF_H__

#include "ros/ros.h"
#include <tf/tf.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <msgs/PointXYZ.h>
#include <msgs/DetectedObjectArray.h>

namespace to_map_tf
{
class ToMapTF
{
public:
  ToMapTF()
  {
  }

  ~ToMapTF()
  {
  }

  int run();

private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_detection_;
  ros::Publisher pub_to_map_tf_;

  tf2_ros::Buffer tf_buffer_;
  std::string frame_id_target_ = "map";
  std::string frame_id_source_ = "lidar";

  void convert(msgs::PointXYZ& p, const geometry_msgs::TransformStamped tf_stamped);
  void callbackDetection(const msgs::DetectedObjectArray::ConstPtr& input);
};
}  // namespace to_map_tf
#endif  // __TO_MAP_TF_H__
