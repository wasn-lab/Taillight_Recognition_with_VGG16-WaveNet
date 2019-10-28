#include <gtest/gtest.h>
#include <cmath>
#include "imu_distance.h"

namespace imu_distance_test
{
inline double roundn(const double x, const int n)
{
  return std::pow(10, -n) * std::round(x * std::pow(10, n));
}

static imu_distance::IMUDistance init_imu_distance()
{
  imu_distance::PoseRPY sensor_pose;
  sensor_pose.position.x = 0.;
  sensor_pose.position.y = 0.;
  sensor_pose.position.z = 0.;

  sensor_pose.orientation.roll_rad = 0.;
  sensor_pose.orientation.pitch_rad = 0.;
  sensor_pose.orientation.yaw_rad = 0.;

  sensor_pose.rpyrate.rollrate_radps = 0.;
  sensor_pose.rpyrate.pitchrate_radps = 0.;
  sensor_pose.rpyrate.yawrate_radps = 0.;

  imu_distance::IMUDistance imud(sensor_pose);

  return imud;
}

TEST(IMUDistanceTest, test_read_obj_bbox2d)
{
  auto imud = init_imu_distance();

  imu_distance::BBox2D b1;
  b1.x = 0;
  b1.y = 0;
  b1.w = 3;
  b1.h = 3;
  imud.read_obj_bbox2d(b1);

  EXPECT_EQ(2., imud.get_obj_pivot_x2d());
  EXPECT_EQ(1., imud.get_obj_pivot_y2d());

  imu_distance::BBox2D b2;
  b2.x = 0;
  b2.y = 0;
  b2.w = 2;
  b2.h = 2;
  imud.read_obj_bbox2d(b2);

  EXPECT_EQ(1., imud.get_obj_pivot_x2d());
  EXPECT_EQ(0.5, imud.get_obj_pivot_y2d());
}

TEST(IMUDistanceTest, test_fov)
{
  EXPECT_EQ(120., roundn(imu_distance::fov_w, 3));
  EXPECT_EQ(75.477, roundn(imu_distance::fov_h, 3));
  EXPECT_EQ(0.063, roundn(imu_distance::degs_per_px, 3));
}

TEST(IMUDistanceTest, test_compute_imu_and_sensor_rpy)
{
  imu_distance::PoseRPY sensor_pose;
  sensor_pose.position.x = 1.;
  sensor_pose.position.y = 0.;
  sensor_pose.position.z = 1.;

  sensor_pose.orientation.roll_rad = 0.1;
  sensor_pose.orientation.pitch_rad = 0.1;
  sensor_pose.orientation.yaw_rad = 0.1;

  sensor_pose.rpyrate.rollrate_radps = 0.;
  sensor_pose.rpyrate.pitchrate_radps = 0.;
  sensor_pose.rpyrate.yawrate_radps = 0.;
  imu_distance::IMUDistance imud(sensor_pose);

  imu_distance::RPYRate imu_rpyrate;
  imu_rpyrate.rollrate_radps = 0.;
  imu_rpyrate.pitchrate_radps = 0.174533;
  imu_rpyrate.yawrate_radps = 0.;
  imud.set_imu_and_sensor_rpyrate(imu_rpyrate, 0.1);

  EXPECT_EQ(0.1, roundn(imud.get_sensor_pose().orientation.roll_rad, 3));
  EXPECT_EQ(0.117, roundn(imud.get_sensor_pose().orientation.pitch_rad, 3));
  EXPECT_EQ(0.1, roundn(imud.get_sensor_pose().orientation.yaw_rad, 3));
}

TEST(IMUDistanceTest, test_pivot2d_to_yaw_pitch)
{
  auto imud = init_imu_distance();

  imu_distance::BBox2D b;
  b.x = 0;
  b.y = 0;
  b.w = 1;
  b.h = 1;
  imud.read_obj_bbox2d(b);

  double obj_yaw_rad_rel = 0.;
  double obj_pitch_rad_rel = 0.;

  imu_distance::x2d_to_yaw(obj_yaw_rad_rel, imud.get_obj_pivot_x2d());
  imu_distance::y2d_to_pitch(obj_pitch_rad_rel, imud.get_obj_pivot_y2d());

  double expect_x_deg = imu_distance::degree_to_radian(0.5 * imu_distance::fov_w);
  double expect_y_deg = imu_distance::degree_to_radian(0.5 * imu_distance::fov_h);
  EXPECT_EQ(roundn(expect_x_deg, 3), roundn(obj_yaw_rad_rel, 3));
  EXPECT_EQ(roundn(-expect_y_deg, 3), roundn(obj_pitch_rad_rel, 3));
}

TEST(IMUDistanceTest, test_compute_sensor2obj_ray_abs)
{
  auto imud = init_imu_distance();

  imud.set_sensor_pose_pitch_rad(0.785398);

  imud.set_sray_to_obj_bottom_m_rel(1., 0., 0.);
  imud.compute_sensor2obj_ray_abs();

  EXPECT_EQ(0.707, roundn(imud.get_sray().to_obj_bottom_m_abs.x, 3));
  EXPECT_EQ(0., roundn(imud.get_sray().to_obj_bottom_m_abs.y, 3));
  EXPECT_EQ(-0.707, roundn(imud.get_sray().to_obj_bottom_m_abs.z, 3));
}

TEST(IMUDistanceTest, test_steps_4_and_5)
{
  auto imud = init_imu_distance();

  imud.set_sensor_position_z(1.925);
  imud.set_sray_to_obj_bottom_m_abs(4., 0., -3.);

  imud.scale_ray_to_touch_ground_abs();
  imud.project_ray_abs_to_ground();

  EXPECT_EQ(4., roundn(imud.get_ego2obj_distance_m(), 3));
}

TEST(IMUDistanceTest, test_whole)
{
  imu_distance::PoseRPY sensor_pose;
  sensor_pose.position.x = 1.;
  sensor_pose.position.y = 0.;
  sensor_pose.position.z = 1.;

  sensor_pose.orientation.roll_rad = 0.;
  sensor_pose.orientation.pitch_rad = 0.785398;
  sensor_pose.orientation.yaw_rad = 0.;

  sensor_pose.rpyrate.rollrate_radps = 0.;
  sensor_pose.rpyrate.pitchrate_radps = 0.;
  sensor_pose.rpyrate.yawrate_radps = 0.;
  imu_distance::IMUDistance imud(sensor_pose);

  imu_distance::RPYRate imu_rpyrate;
  imu_rpyrate.rollrate_radps = 0.;
  imu_rpyrate.pitchrate_radps = -0.261799;
  imu_rpyrate.yawrate_radps = 0.;
  imud.set_imu_and_sensor_rpyrate(imu_rpyrate, 1.);

  imu_distance::BBox2D bbox = (imu_distance::BBox2D){ (unsigned int)std::ceil(0.5 * camera::raw_image_width),
                                                      (unsigned int)std::ceil(0.5 * camera::raw_image_height),
                                                      (unsigned int)1, (unsigned int)1 };
  imud.read_obj_bbox2d(bbox);

  double ego2obj_distance_m = -1.;
  imud.run(ego2obj_distance_m);

  EXPECT_EQ(3.983, roundn(imud.get_ego2obj_distance_m(), 3));
}

};  // namespace
