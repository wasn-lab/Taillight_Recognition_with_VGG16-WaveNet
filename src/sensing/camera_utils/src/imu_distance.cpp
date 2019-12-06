#include "imu_distance.h"

namespace imu_distance
{
void IMUDistance::init(const PoseRPY& sensor_pose)
{
  imu_pose_.position.x = 0.;
  imu_pose_.position.y = 0.;
  imu_pose_.position.z = 0.;

  imu_pose_.orientation.roll_rad = 0.;
  imu_pose_.orientation.pitch_rad = 0.;

  imu_pose_.rpyrate.rollrate_radps = 0.;
  imu_pose_.rpyrate.pitchrate_radps = 0.;
  imu_pose_.rpyrate.yawrate_radps = 0.;

  sensor_pose_.position.x = sensor_pose.position.x;
  sensor_pose_.position.y = sensor_pose.position.y;
  sensor_pose_.position.z = sensor_pose.position.z;

  sensor_pose_.orientation.roll_rad = sensor_pose.orientation.roll_rad;
  sensor_pose_.orientation.pitch_rad = sensor_pose.orientation.pitch_rad;
  sensor_pose_.orientation.yaw_rad = sensor_pose.orientation.yaw_rad;

  sensor_pose_.rpyrate.rollrate_radps = 0.;
  sensor_pose_.rpyrate.pitchrate_radps = 0.;
  sensor_pose_.rpyrate.yawrate_radps = 0.;

  sensor2imu_xz_radius_m_ = std::sqrt(std::pow(sensor_pose_.position.x, 2) + std::pow(sensor_pose_.position.z, 2));
}

void IMUDistance::set_dt_sec(const double dt_sec)
{
  dt_sec_ = dt_sec;
}

void IMUDistance::set_imu_and_sensor_rpyrate(const RPYRate& imu_rpyrate, const double dt_sec)
{
  set_dt_sec(dt_sec);

  imu_pose_.rpyrate.rollrate_radps = imu_rpyrate.rollrate_radps;
  imu_pose_.rpyrate.pitchrate_radps = imu_rpyrate.pitchrate_radps;
  imu_pose_.rpyrate.yawrate_radps = imu_rpyrate.yawrate_radps;

  sensor_pose_.rpyrate.rollrate_radps = imu_rpyrate.rollrate_radps;
  sensor_pose_.rpyrate.pitchrate_radps = imu_rpyrate.pitchrate_radps;
  sensor_pose_.rpyrate.yawrate_radps = imu_rpyrate.yawrate_radps;

  compute_imu_rpy_displacement();
  compute_imu_and_sensor_rpy();
  compute_sensor_position_with_dpitch_only();
}

void IMUDistance::compute_imu_rpy_displacement()
{
  imu_disp.roll_rad = imu_pose_.rpyrate.rollrate_radps * dt_sec_;
  imu_disp.pitch_rad = imu_pose_.rpyrate.pitchrate_radps * dt_sec_;
  imu_disp.yaw_rad = imu_pose_.rpyrate.yawrate_radps * dt_sec_;
}

void IMUDistance::compute_imu_and_sensor_rpy()
{
  imu_pose_.orientation.roll_rad += imu_disp.roll_rad;
  imu_pose_.orientation.pitch_rad += imu_disp.pitch_rad;
  imu_pose_.orientation.yaw_rad += imu_disp.yaw_rad;

  sensor_pose_.orientation.roll_rad += imu_disp.roll_rad;
  sensor_pose_.orientation.pitch_rad += imu_disp.pitch_rad;
  sensor_pose_.orientation.yaw_rad += imu_disp.yaw_rad;
}

void IMUDistance::compute_sensor_position_with_dpitch_only()
{
  glm::dvec3 v(sensor_pose_.position.x, sensor_pose_.position.y, sensor_pose_.position.z);
  glm::dvec3 ky(0., 1., 0.);

  // Rotate 'v', a unit vector on the x-axis 180 degrees
  // around 'ky', a unit vector pointing up on the y-axis.
  glm::dvec3 rotated = rotate(v, ky, imu_disp.pitch_rad);

  sensor_pose_.position.x = rotated.x;  // meter
  sensor_pose_.position.y = rotated.y;  // meter
  sensor_pose_.position.z = rotated.z;  // meter
}

void IMUDistance::read_obj_bbox2d(const BBox2D& bbox)
{
  obj_pivot_x2d_ = (double)(bbox.x + bbox.h - 1);
  obj_pivot_y2d_ = (double)bbox.y + 0.5 * (bbox.w - 1);
}

void IMUDistance::compute_sensor2obj_ray_rel()
{
  sray_.to_obj_bottom_m_rel.x = 0.25 * camera::raw_image_width;  // due to horizontal FOV = 120 degs
  sray_.to_obj_bottom_m_rel.y = 0.5 * camera::raw_image_width - obj_pivot_x2d_;
  sray_.to_obj_bottom_m_rel.z = 0.5 * camera::raw_image_height - obj_pivot_y2d_;
}

void IMUDistance::compute_sensor2obj_ray_abs()
{
  // note:: Here is a simplified assumption that camera's roll and yaw remains
  //        their default value without changed by vehicle shaking
  //        (Not changed according to imu's pitch displacement)

  glm::dvec3 v(sray_.to_obj_bottom_m_rel.x, sray_.to_obj_bottom_m_rel.y, sray_.to_obj_bottom_m_rel.z);
  glm::dvec3 ky(0., 1., 0.);

  // Rotate 'v', a unit vector on the x-axis 180 degrees
  // around 'ky', a unit vector pointing up on the y-axis.
  glm::dvec3 rotated = rotate(v, ky, sensor_pose_.orientation.pitch_rad);

  sray_.to_obj_bottom_m_abs.x = rotated.x;
  sray_.to_obj_bottom_m_abs.y = rotated.y;
  sray_.to_obj_bottom_m_abs.z = rotated.z;
}

void IMUDistance::scale_ray_to_touch_ground_abs()
{
  if (sray_.to_obj_bottom_m_abs.z >= 0.)
  {
    sray_.full_length_m = -1.;  // infinity
  }
  else
  {
    sensor_actual_height_m_ = sensor_pose_.position.z + imu_actual_height_m_;

    double sray_current_length_m =
        std::sqrt(std::pow(sray_.to_obj_bottom_m_abs.x, 2) + std::pow(sray_.to_obj_bottom_m_abs.y, 2) +
                  std::pow(sray_.to_obj_bottom_m_abs.z, 2));

    sray_.full_length_m = sray_current_length_m * std::fabs(sensor_actual_height_m_ / sray_.to_obj_bottom_m_abs.z);
  }
}

void IMUDistance::project_ray_abs_to_ground()
{
  if (sray_.to_obj_bottom_m_abs.z >= 0.)
  {
    ego2obj_distance_m_ = -1.;  // infinity
  }
  else
  {
    glm::dvec3 v1(sray_.to_obj_bottom_m_abs.x, sray_.to_obj_bottom_m_abs.y, sray_.to_obj_bottom_m_abs.z);
    glm::dvec3 v2(0., 0., -1.);
    glm::dvec3 origin(0., 0., 0.);

    double theta_rad = angle_between(v1, v2, origin);

    ego2obj_distance_m_ = sray_.full_length_m * std::sin(theta_rad);
  }
}

int IMUDistance::run(double& ego2obj_distance_m)
{
  if (obj_pivot_x2d_ == -1 || obj_pivot_y2d_ == -1)
  {
    return 1;
  }

  compute_sensor2obj_ray_rel();
  compute_sensor2obj_ray_abs();
  scale_ray_to_touch_ground_abs();
  project_ray_abs_to_ground();

  ego2obj_distance_m = ego2obj_distance_m_;

  return 0;
}

};  // namespace
