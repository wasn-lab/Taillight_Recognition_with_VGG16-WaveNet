#ifndef __VELOCITY_H__
#define __VELOCITY_H__

#include "tpp.h"
#include <stdio.h>  // puts
#include <iostream>
#include <cmath>  // sin, cos
#include "utils.h"

#define UNIT_OF_SPEED 1  // 1: km/h ; 2: m/s

namespace tpp
{
class Velocity
{
public:
  Velocity()
  {
  }
  ~Velocity()
  {
  }

  int init_time(const double secs, const double secs_prev);

  void init_ego_yawrate(const float ego_yawrate);
  void init_ego_speed(const float ego_speed);

  void init_object_relative_position(const float obj_x_rel, const float obj_x_rel_prev,  //
                                     const float obj_y_rel, const float obj_y_rel_prev);

  void compute_ego_position_absolute();

  void compute_position_displacement();

  void update_localization();

  void compute_velocity(msgs::PointXYZV& object_absolute_velocity, msgs::PointXYZV& object_relative_velocity);

  void debug_ego_velocity(const float ego_dx, const float ego_dy, const long long dt);

  void debug_object_velocity(const float ego_dx, const float ego_dy, const float box_center_x, const float box_center_y,
                             const float box_center_x_prev, const float box_center_y_prev, const long long dt,
                             const bool detail);

  // getter
  long long get_dt();
  float get_ego_x_rel();
  float get_ego_y_rel();
  float get_ego_z_rel();
  float get_ego_x_abs();
  float get_ego_y_abs();
  float get_ego_z_abs();
  float get_ego_dx_abs();
  float get_ego_dy_abs();
  float get_ego_dz_abs();
  float get_ego_heading();
  float get_ego_yawrate();
  float get_ego_speed();

  // setter
  void set_dt(const long long dt);
  void set_ego_x_rel(const float ego_x_rel);
  void set_ego_y_rel(const float ego_y_rel);
  void set_ego_z_rel(const float ego_z_rel);
  void set_ego_heading(const float ego_heading);
  void set_ego_yawrate(const float ego_yawrate);
  void set_ego_speed(const float ego_speed);

private:
  DISALLOW_COPY_AND_ASSIGN(Velocity);

#if UNIT_OF_SPEED == 1
  static constexpr unsigned long velocity_mul_ = 3600000000;  // km/h
  const std::string speed_unit_ = "km/h";
#else
  static constexpr unsigned long velocity_mul_ = 1000000000;  // m/s
  const std::string speed_unit_ = "m/s";
#endif

  long long dt_ = 0;
  long long time_ = 0;
  long long time_prev_ = 0;

  float ego_x_abs_ = 0;
  float ego_x_abs_prev_ = 0;
  float ego_x_rel_ = 0;
  float ego_dx_abs_ = 0;
  float ego_dx_rel_ = 0;
  float ego_vx_abs_ = 0;
  float ego_vx_rel_ = 0;

  float ego_y_abs_ = 0;
  float ego_y_abs_prev_ = 0;
  float ego_y_rel_ = 0;
  float ego_dy_abs_ = 0;
  float ego_dy_rel_ = 0;
  float ego_vy_abs_ = 0;
  float ego_vy_rel_ = 0;

  float ego_z_abs_ = 0;
  float ego_z_abs_prev_ = 0;
  float ego_z_rel_ = 0;
  float ego_dz_abs_ = 0;
  float ego_dz_rel_ = 0;
  float ego_vz_abs_ = 0;
  float ego_vz_rel_ = 0;

  float ego_heading_ = 0;
  float ego_heading_prev_ = 0;
  float ego_psi_ = 0;

  float ego_yawrate_ = 0;
  float ego_yawrate_prev_ = 0;

  float ego_speed_ = 0;
  float ego_speed_prev_ = 0;

  float ego_radius_ = 0;

  float obj_x_rel_ = 0;
  float obj_x_rel_prev_ = 0;
  float obj_dx_abs_ = 0;
  float obj_dx_rel_to_each_ego_ = 0;
  float obj_dx_rel_to_prev_ego_ = 0;

  float obj_y_rel_ = 0;
  float obj_y_rel_prev_ = 0;
  float obj_dy_abs_ = 0;
  float obj_dy_rel_to_each_ego_ = 0;
  float obj_dy_rel_to_prev_ego_ = 0;

  float obj_z_rel_ = 0;
  float obj_z_rel_prev_ = 0;
  float obj_dz_abs_ = 0;
  float obj_dz_rel_to_each_ego_ = 0;
  float obj_dz_rel_to_prev_ego_ = 0;

  void compute_velocity_core(msgs::PointXYZV& velocity, const float obj_dx, const float obj_dy);

  void compute_object_relative_position_displacement();

  void compute_object_absolute_position_displacement();
};
}  // namespace tpp

#endif  // __VELOCITY_H__
