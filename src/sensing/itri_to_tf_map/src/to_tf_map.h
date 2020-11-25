#ifndef __to_tf_map_H__
#define __to_tf_map_H__

#include "ros/ros.h"
#include <tf/tf.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <msgs/PointXYZ.h>
#include <msgs/DetectedObjectArray.h>

namespace to_tf_map
{
class ToTFMap
{
public:
  ToTFMap()
  {
  }

  ~ToTFMap()
  {
  }

  int run();

private:
  ros::NodeHandle nh_;

  ros::Subscriber sub_detection_;
  ros::Publisher pub_to_tf_map_;

  std::string in_topic_ = "";
  int num_forecasts_ = 20;

  tf2_ros::Buffer tf_buffer_;
  std::string frame_id_source_ = "base_link";
  std::string frame_id_target_ = "map";

  void convert(msgs::PointXYZ& p, const geometry_msgs::TransformStamped tf_stamped);
  void convert_all_to_map_tf(std::vector<msgs::DetectedObject>& objs);
  void callbackDetection(const msgs::DetectedObjectArray::ConstPtr& input);
};
}  // namespace to_tf_map
#endif  // __to_tf_map_H__
