/*
 * Copyright 2018 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * v1.0 Yukihiro Saito
 */

#include "multi_object_tracker/tracker/model/vehicle_tracker.hpp"
#include "multi_object_tracker/utils/utils.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#define EIGEN_MPL2_ONLY
#include <Eigen/Core>
#include <Eigen/Geometry>

VehicleTracker::VehicleTracker(const ros::Time& time, const autoware_perception_msgs::DynamicObject& object)
  : Tracker(time, object.semantic.type)
  , filtered_yaw_(0.0)
  , yaw_filter_gain_(0.7)
  , is_fixed_yaw_(object.state.orientation_reliable)
  , dim_filter_gain_(0.9)
  , is_fixed_dim_(false)
  , filtered_posx_(object.state.pose_covariance.pose.position.x)
  , filtered_posy_(object.state.pose_covariance.pose.position.y)
  , pos_filter_gain_(0.2)
  , filtered_vx_(0.0)
  , filtered_vy_(0.0)
  , last_filtered_vx_(0.0)
  , last_filtered_vy_(0.0)
  , estimated_ax_(0.0)
  , estimated_ay_(0.0)
  , v_filter_gain_(0.4)
  , area_filter_gain_(0.8)
  , last_measurement_posx_(object.state.pose_covariance.pose.position.x)
  , last_measurement_posy_(object.state.pose_covariance.pose.position.y)
  , last_update_time_(time)
  , last_measurement_time_(time)
{
  object_ = object;
  // yaw
  if (object.shape.type == autoware_perception_msgs::Shape::BOUNDING_BOX)
  {
    double roll, pitch, yaw;
    tf2::Quaternion quaternion;
    tf2::fromMsg(object.state.pose_covariance.pose.orientation, quaternion);
    tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);
    yaw = std::atan2(std::sin(yaw), std::cos(yaw));
    filtered_yaw_ = yaw;
  }
  // dim
  if (object.shape.type == autoware_perception_msgs::Shape::BOUNDING_BOX)
  {
    is_fixed_dim_ = true;
    filtered_dim_x_ = object.shape.dimensions.x;
    filtered_dim_y_ = object.shape.dimensions.y;
  }
  // area
  filtered_area_ = utils::getArea(object.shape);
}

bool VehicleTracker::predict(const ros::Time& time)
{
  double dt = (time - last_update_time_).toSec();
  if (dt < 0.0)
    dt = 0.0;
  double vel = std::cos(filtered_yaw_) * filtered_vx_ + std::sin(filtered_yaw_) * filtered_vy_;
  if (vel < 0.0 && !is_fixed_yaw_)
  {
    filtered_posx_ += std::cos(filtered_yaw_ + M_PI) * std::fabs(vel) * dt;
    filtered_posy_ += std::sin(filtered_yaw_ + M_PI) * std::fabs(vel) * dt;
    // filtered_posx_ += filtered_vx_ * dt;
    // filtered_posy_ += filtered_vy_ * dt;
  }
  else
  {
    filtered_posx_ += std::cos(filtered_yaw_) * vel * dt;
    filtered_posy_ += std::sin(filtered_yaw_) * vel * dt;
    // filtered_posx_ += filtered_vx_ * dt;
    // filtered_posy_ += filtered_vy_ * dt;
  }
  last_update_time_ = time;
  return true;
}
bool VehicleTracker::measure(const autoware_perception_msgs::DynamicObject& object, const ros::Time& time)
{
  int type = object.semantic.type;
  bool is_changed_unknown_object = false;
  if (type == autoware_perception_msgs::Semantic::UNKNOWN)
  {
    type = getType();
    is_changed_unknown_object = true;
  }
  object_ = object;
  setType(type);

  // yaw
  if (object.shape.type == autoware_perception_msgs::Shape::BOUNDING_BOX)
  {
    double roll, pitch, yaw;
    tf2::Quaternion quaternion;
    tf2::fromMsg(object.state.pose_covariance.pose.orientation, quaternion);
    tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);
    yaw = std::atan2(std::sin(yaw), std::cos(yaw));
    if (!is_fixed_yaw_ && object.state.orientation_reliable)
    {
      if (Eigen::Vector2d(std::cos(filtered_yaw_), std::sin(filtered_yaw_))
              .dot(Eigen::Vector2d(std::cos(yaw), std::sin(yaw))) < 0.0)
        filtered_yaw_ += M_PI;
    }
    else if (!is_fixed_yaw_ && !object.state.orientation_reliable)
    {
      if (Eigen::Vector2d(std::cos(filtered_yaw_), std::sin(filtered_yaw_))
              .dot(Eigen::Vector2d(std::cos(yaw), std::sin(yaw))) < 0.0)
        yaw += M_PI;
    }
    else if (is_fixed_yaw_ && object.state.orientation_reliable)
    {
    }
    else if (is_fixed_yaw_ && !object.state.orientation_reliable)
    {
      if (Eigen::Vector2d(std::cos(filtered_yaw_), std::sin(filtered_yaw_))
              .dot(Eigen::Vector2d(std::cos(yaw), std::sin(yaw))) < 0.0)
        yaw += M_PI;
    }

    if (object.state.orientation_reliable)
      is_fixed_yaw_ = object.state.orientation_reliable;

    Eigen::Vector2d filtered_yaw_vector =
        yaw_filter_gain_ * Eigen::Vector2d(std::cos(filtered_yaw_), std::sin(filtered_yaw_)) +
        (1.0 - yaw_filter_gain_) * Eigen::Vector2d(std::cos(yaw), std::sin(yaw));
    filtered_yaw_ = std::atan2(filtered_yaw_vector.y(), filtered_yaw_vector.x());
  }
  // dim
  if (object.shape.type == autoware_perception_msgs::Shape::BOUNDING_BOX)
  {
    if (!is_fixed_dim_)
    {
      filtered_dim_x_ = object.shape.dimensions.x;
      filtered_dim_y_ = object.shape.dimensions.y;
      is_fixed_dim_ = true;
    }
    else
    {
      filtered_dim_x_ = dim_filter_gain_ * filtered_dim_x_ + (1.0 - dim_filter_gain_) * object.shape.dimensions.x;
      filtered_dim_y_ = dim_filter_gain_ * filtered_dim_y_ + (1.0 - dim_filter_gain_) * object.shape.dimensions.y;
    }
  }
  // area
  filtered_area_ = area_filter_gain_ * filtered_area_ + (1.0 - area_filter_gain_) * utils::getArea(object.shape);

  // vx,vy
  double dt = (time - last_measurement_time_).toSec();
  last_measurement_time_ = time;
  last_update_time_ = time;
  if (0.0 < dt)
  {
    double current_vel = std::sqrt((object.state.pose_covariance.pose.position.x - last_measurement_posx_) *
                                       (object.state.pose_covariance.pose.position.x - last_measurement_posx_) +
                                   (object.state.pose_covariance.pose.position.y - last_measurement_posy_) *
                                       (object.state.pose_covariance.pose.position.y - last_measurement_posy_));
    const double max_vel = 20.0; /* [m/s]*/
    const double vel_scale = max_vel < current_vel ? max_vel / current_vel : 1.0;
    if (is_changed_unknown_object)
    {
      filtered_vx_ =
          0.95 * filtered_vx_ +
          (1.0 - 0.95) * ((object.state.pose_covariance.pose.position.x - last_measurement_posx_) / dt) * vel_scale;
      filtered_vy_ =
          0.95 * filtered_vy_ +
          (1.0 - 0.95) * ((object.state.pose_covariance.pose.position.y - last_measurement_posy_) / dt) * vel_scale;
    }
    else
    {
      filtered_vx_ = v_filter_gain_ * filtered_vx_ +
                     (1.0 - v_filter_gain_) *
                         ((object.state.pose_covariance.pose.position.x - last_measurement_posx_) / dt) * vel_scale;
      filtered_vy_ = v_filter_gain_ * filtered_vy_ +
                     (1.0 - v_filter_gain_) *
                         ((object.state.pose_covariance.pose.position.y - last_measurement_posy_) / dt) * vel_scale;
      v_filter_gain_ = std::min(0.9, v_filter_gain_ + 0.15);
    }

    double vel = std::cos(filtered_yaw_) * filtered_vx_ + std::sin(filtered_yaw_) * filtered_vy_;
    filtered_vx_ = std::cos(filtered_yaw_) * vel;
    filtered_vy_ = std::sin(filtered_yaw_) * vel;

    // estimate acceleration
    estimated_ax_ = (filtered_vx_ - last_filtered_vx_) / dt;
    estimated_ay_ = (filtered_vy_ - last_filtered_vy_) / dt;
    last_filtered_vx_ = filtered_vx_;
    last_filtered_vy_ = filtered_vy_;
  }

  // pos x, pos y
  last_measurement_posx_ = object.state.pose_covariance.pose.position.x;
  last_measurement_posy_ = object.state.pose_covariance.pose.position.y;
  filtered_posx_ =
      pos_filter_gain_ * filtered_posx_ + (1.0 - pos_filter_gain_) * object.state.pose_covariance.pose.position.x;
  filtered_posy_ =
      pos_filter_gain_ * filtered_posy_ + (1.0 - pos_filter_gain_) * object.state.pose_covariance.pose.position.y;
  pos_filter_gain_ = std::min(0.8, pos_filter_gain_ + 0.05);

  return true;
}

bool VehicleTracker::getEstimatedDynamicObject(const ros::Time& time, autoware_perception_msgs::DynamicObject& object)
{
  object = object_;
  object.id = unique_id::toMsg(getUUID());
  object.semantic.type = getType();

  // fill: heading
  if (object.shape.type == autoware_perception_msgs::Shape::BOUNDING_BOX)
  {
    double roll, pitch, yaw;
    tf2::Quaternion quaternion;
    tf2::fromMsg(object.state.pose_covariance.pose.orientation, quaternion);
    tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);
    tf2::Quaternion filtered_quaternion;
    filtered_quaternion.setRPY(roll, pitch, filtered_yaw_);
    object.state.pose_covariance.pose.orientation = tf2::toMsg(filtered_quaternion);

    object.state.orientation_reliable = is_fixed_yaw_;
  }

  // fill: dimension
  if (is_fixed_dim_ && object.shape.type == autoware_perception_msgs::Shape::BOUNDING_BOX)
  {
    object.shape.dimensions.x = filtered_dim_x_;
    object.shape.dimensions.y = filtered_dim_y_;
  }

  // fill: position
  object.state.pose_covariance.pose.position.x = filtered_posx_;
  object.state.pose_covariance.pose.position.y = filtered_posy_;

  double dt = (time - last_update_time_).toSec();
  if (dt < 0.0)
    dt = 0.0;

  double vel = std::cos(filtered_yaw_) * filtered_vx_ + std::sin(filtered_yaw_) * filtered_vy_;

  if (vel < 0.0 && !is_fixed_yaw_)
  {
    object.state.pose_covariance.pose.position.x += std::cos(filtered_yaw_ + M_PI) * std::fabs(vel) * dt;
    object.state.pose_covariance.pose.position.y += std::sin(filtered_yaw_ + M_PI) * std::fabs(vel) * dt;
  }
  else
  {
    object.state.pose_covariance.pose.position.x += std::cos(filtered_yaw_) * std::max(vel, 0.0) * dt;
    object.state.pose_covariance.pose.position.y += std::sin(filtered_yaw_) * std::max(vel, 0.0) * dt;
  }

  // fill: velocity
  // IMPORTANT:
  // twist.linear.x and .y:  base_link-frame filtered_vx_ filtered_vy_
  // twist.angular.x and .y:       map-frame filtered_vx_ filtered_vy_
  // ----------------------
  // The standard way: Place map-frame filtered_vx_ filtered_vy_ on twist.linear.x and .y, respectively,
  // since the frame_id of /objects is set 'map'.
  // However, Tier4 designed twist.linear.x and .y in this fashion for map_based_prediction's use, and
  // this hindered us from adjusting them freely.

  double roll, pitch, yaw;
  tf2::Quaternion quaternion;
  tf2::fromMsg(object.state.pose_covariance.pose.orientation, quaternion);
  tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);

  if (object.shape.type == autoware_perception_msgs::Shape::BOUNDING_BOX)
  {
    object.state.twist_covariance.twist.linear.x = filtered_vx_ * std::cos(-yaw) - filtered_vy_ * std::sin(-yaw);
    object.state.twist_covariance.twist.linear.y = 0.0;
  }
  else
  {
    object.state.twist_covariance.twist.linear.x = filtered_vx_ * std::cos(-yaw) - filtered_vy_ * std::sin(-yaw);
    object.state.twist_covariance.twist.linear.y = filtered_vx_ * std::sin(-yaw) + filtered_vy_ * std::cos(-yaw);
  }

  object.state.twist_covariance.twist.angular.x = filtered_vx_;
  object.state.twist_covariance.twist.angular.y = filtered_vy_;

  // cut microspeed
  constexpr float min_velocity = 0.56;  // 2.0 kmph
  if (std::fabs(object.state.twist_covariance.twist.linear.x) < min_velocity)
  {
    object.state.twist_covariance.twist.linear.x = 0.0;
  }
  if (std::fabs(object.state.twist_covariance.twist.linear.y) < min_velocity)
  {
    object.state.twist_covariance.twist.linear.y = 0.0;
  }

  object.state.twist_reliable = true;

  // fill: acceleration
  object.state.acceleration_covariance.accel.linear.x = estimated_ax_;
  object.state.acceleration_covariance.accel.linear.y = estimated_ay_;

  object.state.acceleration_reliable = true;

  return true;
}
