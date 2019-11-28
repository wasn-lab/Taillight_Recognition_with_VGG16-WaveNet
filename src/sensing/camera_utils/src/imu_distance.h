#ifndef __IMU_DISTANCE_H__
#define __IMU_DISTANCE_H__

#include "imu_utils.h"
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>

namespace imu_distance
{
struct RPY
{
  double roll_rad = 0.;
  double pitch_rad = 0.;
  double yaw_rad = 0.;
};

struct RPYRate
{
  double rollrate_radps = 0.;
  double pitchrate_radps = 0.;
  double yawrate_radps = 0.;
};

struct PoseRPY
{
  geometry_msgs::Point position;
  RPY orientation;
  RPYRate rpyrate;
};

struct BBox2D
{
  unsigned int x;
  unsigned int y;
  unsigned int w;
  unsigned int h;
};

struct SensorRay
{
  geometry_msgs::Vector3 to_obj_bottom_m_rel;  // abs: based on sensor's coordinate
  geometry_msgs::Vector3 to_obj_bottom_m_abs;  // abs: based on imu's coordinate
  double full_length_m;                        // scaled vector which touchs ground (z = 0);
};

class IMUDistance
{
public:
  IMUDistance(const PoseRPY& sensor_pose) : sray_()
  {
    init(sensor_pose);
  }

  ~IMUDistance()
  {
  }

  PoseRPY& get_sensor_pose()
  {
    return sensor_pose_;
  };

  void set_sensor_pose_pitch_rad(const double rad)
  {
    sensor_pose_.orientation.pitch_rad = rad;
  }

  void set_sensor_position_z(const double z)
  {
    sensor_pose_.position.z = z;
  }

  void set_sray_to_obj_bottom_m_rel(const double x, const double y, const double z)
  {
    sray_.to_obj_bottom_m_rel.x = x;
    sray_.to_obj_bottom_m_rel.y = y;
    sray_.to_obj_bottom_m_rel.z = z;
  }

  void set_sray_to_obj_bottom_m_abs(const double x, const double y, const double z)
  {
    sray_.to_obj_bottom_m_abs.x = x;
    sray_.to_obj_bottom_m_abs.y = y;
    sray_.to_obj_bottom_m_abs.z = z;
  }

  SensorRay& get_sray()
  {
    return sray_;
  };

  double& get_dt_sec()
  {
    return dt_sec_;
  }

  double& get_obj_pivot_x2d()
  {
    return obj_pivot_x2d_;
  }

  double& get_obj_pivot_y2d()
  {
    return obj_pivot_y2d_;
  }

  double& get_ego2obj_distance_m()
  {
    return ego2obj_distance_m_;
  }

  // step 1
  void set_dt_sec(const double dt_sec);
  void set_imu_and_sensor_rpyrate(const RPYRate& imu_rpyrate, const double dt_sec);

  // step 2
  void read_obj_bbox2d(const BBox2D& bbox);

  // step 3 ~ 5
  int run(double& ego2obj_distance_m);

  // step 3.1
  void compute_sensor2obj_ray_rel();
  // step 3.2
  void compute_sensor2obj_ray_abs();
  // step 4
  void scale_ray_to_touch_ground_abs();
  // step 5
  void project_ray_abs_to_ground();

private:
  PoseRPY imu_pose_;
  PoseRPY sensor_pose_;
  SensorRay sray_;

  double sensor2imu_xz_radius_m_ = 0.;

  double dt_sec_ = 0.;

  RPY imu_disp;

  double obj_pivot_x2d_ = -1.;
  double obj_pivot_y2d_ = -1.;

  const double imu_actual_height_m_ = 1.075;
  double sensor_actual_height_m_ = 0.;

  double ego2obj_distance_m_ = 0.;

  void init(const PoseRPY& sensor_pose);

  // step 1.1
  void compute_imu_rpy_displacement();
  // step 1.2
  void compute_imu_and_sensor_rpy();
  // step 1.3
  void compute_sensor_position_with_dpitch_only();
};

};  // namespace

#endif  // __IMU_DISTANCE_H__
