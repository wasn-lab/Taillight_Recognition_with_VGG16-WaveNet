#ifndef __XYZ2LLA_H__
#define __XYZ2LLA_H__

#include "geodetic_converter.h"
#include "ros/ros.h"
#include <ros/package.h>
#include <fstream>
#include <tf/tf.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <std_msgs/Empty.h>
#include <msgs/DetectedObjectArray.h>

#include "gnss_utility/gnss_utility.h"
#include "gnss_utility_utm/gnss_utility_utm.h"

#define TWD97 0
#define UTM 1
#define DEBUG 0
#define HEARTBEAT 1

namespace xyz2lla
{
class XYZ2LLA
{
public:
  XYZ2LLA()
  {
  }

  ~XYZ2LLA()
  {
  }

  int run();

private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_xyz2lla_;
  ros::Publisher pub_xyz2lla_;
#if HEARTBEAT == 1
  ros::Publisher pub_xyz2lla_heartbeat_;
#endif

  tf2_ros::Buffer tf_buffer_;
  std::string frame_id_target_ = "map";
  std::string frame_id_source_ = "lidar";

  std::string root_ = "itri_xyz2lla";

  double lat0_ = 0.;  // initial latitude
  double lon0_ = 0.;  // initial longitude
  double alt0_ = 0.;  // initial altitude
  double R_[3][3];    // XYZ to ENU R
  double T_[3];       // XYZ to ENU T
  double T1_[3];
  double T2_[3];
  double T3_[3];
  double T4_[3];
  double T5_[3];
  double R1_[3][3];
  double R2_[3][3];
  double R3_[3][3];
  double R4_[3][3];
  double R5_[3][3];

  bool XYZ2ENU_switch_ = true;

  // twd97-to-wgs84
  double twd97_shift_x_;
  double twd97_shift_y_;
  double twd97_shift_z_;

  // utm-to-wgs84
  double utm_shift_x_;
  double utm_shift_y_;
  int utm_zone_;

  gnss_utility::gnss gnss_tf_;
  gnss_utility_utm::gnss_utm gnss_utm_tf_;
  geodetic_converter::GeodeticConverter geodetic_converter_;

  void testGNSS();

  void initParam();
  void initParamTWD97();
  void initParamUTM();

  void convert(double& out_lat_wgs84, double& out_lon_wgs84, double& out_alt_wgs84, double& out_E, double& out_N,
               double& out_U, const double in_x, const double in_y, const double in_z);
  void callbackTracking(const msgs::DetectedObjectArray::ConstPtr& input);
};
}  // namespace xyz2lla
#endif  // __XYZ2LLA_H__
