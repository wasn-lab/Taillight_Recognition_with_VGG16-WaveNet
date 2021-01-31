#ifndef __DAF_NODE_H__
#define __DAF_NODE_H__

#include "daf.h"
#include <visualization_msgs/MarkerArray.h>
#include <tf2/utils.h>
#include <geometry_msgs/PolygonStamped.h>

#include <fstream>

namespace daf
{
class DAFNode
{
public:
  DAFNode()
  {
  }

  ~DAFNode()
  {
  }

  int run();

private:
  int input_source_ = InputSource::LidarDet;

  geometry_msgs::TransformStamped tf_stamped_;

  std_msgs::Header objs_header_;
  std::vector<msgs::DetectedObject> objs_;

  std::string frame_id_source_ = "base_link";
  std::string frame_id_target_ = "map";

  ros::NodeHandle nh_;
  ros::NodeHandle nh2_;

  // custom callback queue
  ros::CallbackQueue queue_;

  ros::Publisher pub_daf_;
  ros::Publisher pub_daf_heartbeat_;
  ros::Publisher pub_drivable_area_;

  ros::Subscriber sub_input_;
  void callback_detected_objs(const msgs::DetectedObjectArray::ConstPtr& input);

  tf2_ros::Buffer tf_buffer_;

  ros::Subscriber sub_lanelet2_route_;
  void callback_lanelet2_route(const visualization_msgs::MarkerArray::ConstPtr& input);

  bool drivable_area_filter_ = true;
  double expand_left_ = 2.2;
  double expand_right_ = 0.0;
  double ground_z_ = -3.1;

  std::vector<cv::Point3f> lanelet2_route_left;
  std::vector<cv::Point3f> lanelet2_route_right;

  std::vector<cv::Point2f> expanded_route_left;
  std::vector<cv::Point2f> expanded_route_right;

  void get_current_ego_data(const ros::Time input_stamp);

  geometry_msgs::Point get_transform_coordinate(geometry_msgs::Point origin_point, double yaw,
                                                geometry_msgs::Vector3 translation);
  bool check_in_polygon(cv::Point2f position, std::vector<cv::Point2f>& polygon);
  bool drivable_area_filter(const msgs::BoxPoint box_point);

  void set_ros_params();
  void subscribe_and_advertise_topics();
  void publish_filtered_objs();
};
}  // namespace daf

#endif  // __DAF_NODE_H__
