#include "velocity.h"

namespace tpp
{
// === getter ===

long long Velocity::get_dt()
{
  return dt_;
}

float Velocity::get_ego_x_rel()
{
  return ego_x_rel_;
}

float Velocity::get_ego_y_rel()
{
  return ego_y_rel_;
}

float Velocity::get_ego_z_rel()
{
  return ego_z_rel_;
}

float Velocity::get_ego_x_abs()
{
  return ego_x_abs_;
}

float Velocity::get_ego_y_abs()
{
  return ego_y_abs_;
}

float Velocity::get_ego_z_abs()
{
  return ego_z_abs_;
}

float Velocity::get_ego_dx_abs()
{
  return ego_dx_abs_;
}

float Velocity::get_ego_dy_abs()
{
  return ego_dy_abs_;
}

float Velocity::get_ego_dz_abs()
{
  return ego_dz_abs_;
}

float Velocity::get_ego_heading()
{
  return ego_heading_;
}

float Velocity::get_ego_yawrate()
{
  return ego_yawrate_;
}

float Velocity::get_ego_speed()
{
  return ego_speed_;
}

// === setter ===

void Velocity::set_dt(const long long dt)
{
  dt_ = dt;
}

void Velocity::set_ego_x_rel(const float ego_x_rel)
{
  ego_x_rel_ = ego_x_rel;
}

void Velocity::set_ego_y_rel(const float ego_y_rel)
{
  ego_y_rel_ = ego_y_rel;
}

void Velocity::set_ego_heading(const float ego_heading)
{
  ego_heading_ = ego_heading;
}

void Velocity::set_ego_yawrate(const float ego_yawrate)
{
  ego_yawrate_ = ego_yawrate;
}

void Velocity::set_ego_speed(const float ego_speed)
{
  ego_speed_ = ego_speed;
}

// ==============

int Velocity::init_time(const double secs, const double secs_prev)
{
  time_ = secs * 1000000000;            // nanoseconds
  time_prev_ = secs_prev * 1000000000;  // nanoseconds

  dt_ = time_ - time_prev_;  // nanoseconds

  if (dt_ == 0)
  {
    dt_ = 100000000;
  }
  else if (dt_ < 0)
  {
#if DEBUG_COMPACT
    LOG_INFO << "Warning: dt = " << (dt_ / 1000000.0) << "ms ! Illegal time input !" << std::endl;

    LOG_INFO << "time t-1: " << time_prev_ << std::endl;
    LOG_INFO << "time t  : " << time_ << std::endl;
#endif

    return 1;
  }

#if DEBUG_COMPACT
  LOG_INFO << "dt = " << (dt_ / 1000000.0) << " ms" << std::endl;
#endif

  return 0;
}

void Velocity::init_ego_yawrate(const float ego_yawrate)
{
  ego_yawrate_prev_ = ego_yawrate_;
  ego_yawrate_ = ego_yawrate;
}

void Velocity::init_ego_speed(const float ego_speed)
{
  ego_speed_prev_ = ego_speed_;
  ego_speed_ = ego_speed;
}

void Velocity::init_object_relative_position(const float obj_x_rel, const float obj_x_rel_prev,  //
                                             const float obj_y_rel, const float obj_y_rel_prev)
{
  obj_x_rel_ = obj_x_rel;
  obj_x_rel_prev_ = obj_x_rel_prev;

  obj_y_rel_ = obj_y_rel;
  obj_y_rel_prev_ = obj_y_rel_prev;
}

void Velocity::compute_ego_position_absolute()
{
  ego_x_abs_prev_ = ego_x_abs_;  // meter
  ego_y_abs_prev_ = ego_y_abs_;  // meter

  float buf[1][2] = { { ego_x_rel_, ego_y_rel_ } };
  rotate(buf, 1, ego_heading_);

  ego_x_abs_ = buf[0][0];  // meter
  ego_y_abs_ = buf[0][1];  // meter

  ego_dx_abs_ = ego_x_abs_ - ego_x_abs_prev_;  // meters
  ego_dy_abs_ = ego_y_abs_ - ego_y_abs_prev_;  // meters
}

void Velocity::compute_position_displacement()
{
  ego_psi_ = ego_yawrate_prev_ * dt_ / 1000000000;  // angular displacement

  if (ego_yawrate_prev_ == 0)
  {
    ego_dx_rel_ = ego_speed_prev_ * dt_ / 1000000000;  // meter
    ego_dy_rel_ = 0;                                   // meter
    return;
  }

  ego_radius_ = ego_speed_prev_ / ego_yawrate_prev_;  // meter

  ego_dx_rel_ = ego_radius_ * std::sin(ego_psi_);        // meters
  ego_dy_rel_ = ego_radius_ * (1 - std::cos(ego_psi_));  // meters
}

void Velocity::update_localization()
{
  ego_x_abs_prev_ = ego_x_abs_;  // meter
  ego_y_abs_prev_ = ego_y_abs_;  // meter

  ego_heading_prev_ = ego_heading_;

  float buf[1][2] = { { ego_dx_rel_, ego_dy_rel_ } };
  rotate(buf, 1, ego_heading_prev_);

  ego_dx_abs_ = buf[0][0];
  ego_dy_abs_ = buf[0][1];

  ego_x_abs_ += ego_dx_abs_;  // meter
  ego_y_abs_ += ego_dy_abs_;  // meter

  ego_heading_ += ego_psi_;  // radian
}

void Velocity::compute_object_relative_position_displacement()
{
  obj_dx_rel_to_each_ego_ = obj_x_rel_ - obj_x_rel_prev_;  // meter
  obj_dy_rel_to_each_ego_ = obj_y_rel_ - obj_y_rel_prev_;  // meter
}

void Velocity::compute_object_absolute_position_displacement()
{
  obj_dx_rel_to_prev_ego_ = ego_dx_rel_ + obj_dx_rel_to_each_ego_;  // meter
  obj_dy_rel_to_prev_ego_ = ego_dy_rel_ + obj_dy_rel_to_each_ego_;  // meter

  float buf[1][2] = { { obj_dx_rel_to_prev_ego_, obj_dy_rel_to_prev_ego_ } };
  rotate(buf, 1, ego_heading_prev_);

  obj_dx_abs_ = buf[0][0];  // meter
  obj_dy_abs_ = buf[0][1];  // meter
}

void Velocity::compute_velocity_core(msgs::PointXYZV& velocity, const float obj_dx, const float obj_dy)
{
  velocity.x = (obj_dx * velocity_mul_) / dt_;
  velocity.y = (obj_dy * velocity_mul_) / dt_;
  velocity.z = 0;
  velocity.speed = euclidean_distance(velocity.x, velocity.y);
}

void Velocity::compute_velocity(msgs::PointXYZV& object_absolute_velocity, msgs::PointXYZV& object_relative_velocity)
{
  obj_dx_rel_to_each_ego_ = 0;  // meter
  obj_dy_rel_to_each_ego_ = 0;  // meter

  compute_object_relative_position_displacement();
  compute_velocity_core(object_relative_velocity, obj_dx_rel_to_each_ego_, obj_dy_rel_to_each_ego_);

  obj_dx_abs_ = 0;  // meter
  obj_dy_abs_ = 0;  // meter

  compute_object_absolute_position_displacement();
  compute_velocity_core(object_absolute_velocity, obj_dx_abs_, obj_dy_abs_);
}

void Velocity::debug_ego_velocity(const float ego_dx, const float ego_dy, const long long dt)
{
  float ego_vx = (ego_dx * velocity_mul_) / dt;
  float ego_vy = (ego_dy * velocity_mul_) / dt;
  float ego_speed = euclidean_distance(ego_vx, ego_vy);

  LOG_INFO << "dt " << dt / 1000000.0f << " ms" << std::endl;

  LOG_INFO << "ego_dx " << ego_dx << std::endl;
  LOG_INFO << "ego_vx " << ego_vx << " " << speed_unit_ << std::endl;

  LOG_INFO << "ego_dy " << ego_dy << std::endl;
  LOG_INFO << "ego_vy " << ego_vy << " " << speed_unit_ << std::endl;

  LOG_INFO << "ego_speed " << ego_speed << " " << speed_unit_ << std::endl;
}

void Velocity::debug_object_velocity(const float ego_dx, const float ego_dy, const float box_center_x,
                                     const float box_center_y, const float box_center_x_prev,
                                     const float box_center_y_prev, const long long dt, const bool detail)
{
  LOG_INFO << "dt " << dt / 1000000.0f << " ms" << std::endl;

  if (detail)
    LOG_INFO << "ego_dx " << ego_dx << " box_center: x " << box_center_x << " x_prev " << box_center_x_prev << " dx "
             << box_center_x - box_center_x_prev << std::endl;
  else
    LOG_INFO << "ego_dx " << ego_dx << " box_center: dx " << box_center_x - box_center_x_prev << std::endl;

  if (detail)
    LOG_INFO << "ego_dy " << ego_dy << " box_center: y " << box_center_y << " y_prev " << box_center_y_prev << " dy "
             << box_center_y - box_center_y_prev << std::endl;
  else
    LOG_INFO << "ego_dy " << ego_dy << " box_center: dy " << box_center_y - box_center_y_prev << std::endl;
}
}  // namespace tpp
