/*
 * Copyright 2020 Tier IV, Inc. All rights reserved.
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
 */
#include <scene_module/crosswalk/scene_walkway.h>
#include <utilization/util.h>

#include <cmath>

namespace bg = boost::geometry;
using Point = bg::model::d2::point_xy<double>;
using Polygon = bg::model::polygon<Point, false>;
using Line = bg::model::linestring<Point>;

WalkwayModule::WalkwayModule(
  const int64_t module_id, const lanelet::ConstLanelet & walkway,
  const PlannerParam & planner_param)
: SceneModuleInterface(module_id), module_id_(module_id), walkway_(walkway), state_(State::APPROACH)
{
  planner_param_ = planner_param;
}

bool WalkwayModule::modifyPathVelocity(
  autoware_planning_msgs::PathWithLaneId * path, autoware_planning_msgs::StopReason * stop_reason)
{
  debug_data_ = {};
  debug_data_.base_link2front = planner_data_->base_link2front;
  first_stop_path_point_index_ = static_cast<int>(path->points.size()) - 1;
  *stop_reason = planning_utils::initializeStopReason(autoware_planning_msgs::StopReason::WALKWAY);

  const auto input = *path;

  if (state_ == State::APPROACH) {
    // create polygon
    lanelet::CompoundPolygon3d lanelet_polygon = walkway_.polygon3d();
    Polygon polygon;
    for (const auto & lanelet_point : lanelet_polygon) {
      polygon.outer().push_back(bg::make<Point>(lanelet_point.x(), lanelet_point.y()));
    }
    polygon.outer().push_back(polygon.outer().front());

    if (!insertTargetVelocityPoint(
          input, polygon, planner_param_.stop_margin, 0.0, *planner_data_, *path, debug_data_,
          first_stop_path_point_index_))
      return false;

    // update state
    const Point self_pose = {
      planner_data_->current_pose.pose.position.x, planner_data_->current_pose.pose.position.y};
    const double distance = bg::distance(polygon, self_pose);
    const double distance_threshold =
      planner_param_.stop_margin + planner_data_->base_link2front + 1.0;
    if (distance < distance_threshold && planner_data_->isVehicleStopping()) state_ = State::STOP;
    return true;
  } else if (state_ == State::STOP) {
    /* get stop point and stop factor */
    autoware_planning_msgs::StopFactor stop_factor;
    stop_factor.stop_pose = debug_data_.first_stop_pose;
    stop_factor.stop_factor_points.emplace_back(debug_data_.nearest_collision_point);
    planning_utils::appendStopReason(stop_factor, stop_reason);

    if (planner_data_->isVehicleStopping()) {
      state_ = State::SURPASSED;
    }
    return true;
  } else if (state_ == State::SURPASSED) {
    return true;
  }
}
