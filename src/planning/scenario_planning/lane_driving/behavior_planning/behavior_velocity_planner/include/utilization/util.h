/*
 * Copyright 2015-2019 Autoware Foundation. All rights reserved.
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
 */

#ifndef COMMON_MATH_PLANNING_UTILS_H
#define COMMON_MATH_PLANNING_UTILS_H

#include <autoware_perception_msgs/DynamicObjectArray.h>
#include <autoware_planning_msgs/Path.h>
#include <autoware_planning_msgs/PathWithLaneId.h>
#include <autoware_planning_msgs/StopReason.h>
#include <autoware_planning_msgs/Trajectory.h>
#include <autoware_planning_msgs/TrajectoryPoint.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <pcl_ros/point_cloud.h>
#include <tf2/utils.h>
#include <visualization_msgs/Marker.h>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <vector>

using Point2d = boost::geometry::model::d2::point_xy<double>;
namespace planning_utils
{
inline geometry_msgs::Point getPoint(const geometry_msgs::Point & p) { return p; }
inline geometry_msgs::Point getPoint(const geometry_msgs::Pose & p) { return p.position; }
inline geometry_msgs::Point getPoint(const geometry_msgs::PoseStamped & p)
{
  return p.pose.position;
}
inline geometry_msgs::Point getPoint(const autoware_planning_msgs::PathPoint & p)
{
  return p.pose.position;
}
inline geometry_msgs::Point getPoint(const autoware_planning_msgs::PathPointWithLaneId & p)
{
  return p.point.pose.position;
}
inline geometry_msgs::Point getPoint(const autoware_planning_msgs::TrajectoryPoint & p)
{
  return p.pose.position;
}
inline geometry_msgs::Pose getPose(const autoware_planning_msgs::Path & path, int idx)
{
  return path.points.at(idx).pose;
}
inline geometry_msgs::Pose getPose(const autoware_planning_msgs::PathWithLaneId & path, int idx)
{
  return path.points.at(idx).point.pose;
}
inline geometry_msgs::Pose getPose(const autoware_planning_msgs::Trajectory & traj, int idx)
{
  return traj.points.at(idx).pose;
}

inline int64_t bitShift(int64_t original_id) { return (original_id << (sizeof(int32_t) * 8 / 2)); }

inline double square(const double & a) { return a * a; }
double normalizeEulerAngle(double euler);
geometry_msgs::Quaternion getQuaternionFromYaw(double yaw);

template <class T1, class T2>
double calcSquaredDist2d(const T1 & a, const T2 & b)
{
  return square(getPoint(a).x - getPoint(b).x) + square(getPoint(a).y - getPoint(b).y);
}

template <class T1, class T2>
double calcDist2d(const T1 & a, const T2 & b)
{
  return std::sqrt(calcSquaredDist2d<T1, T2>(a, b));
}

template <class T>
bool calcClosestIndex(
  const T & path, const geometry_msgs::Pose & pose, int & closest, double dist_thr = 3.0,
  double angle_thr = M_PI_4);

template <class T>
bool calcClosestIndex(
  const T & path, const geometry_msgs::Point & point, int & closest, double dist_thr = 3.0);

geometry_msgs::Pose transformRelCoordinate2D(
  const geometry_msgs::Pose & target, const geometry_msgs::Pose & origin);
geometry_msgs::Pose transformAbsCoordinate2D(
  const geometry_msgs::Pose & relative, const geometry_msgs::Pose & origin);

double calcJudgeLineDist(
  const double velocity, const double max_stop_acceleration, const double delay_response_time);

autoware_planning_msgs::StopReason initializeStopReason(const std::string & stop_reason);
void appendStopReason(
  const autoware_planning_msgs::StopFactor stop_factor,
  autoware_planning_msgs::StopReason * stop_reason);

std::vector<geometry_msgs::Point> toRosPoints(
  const autoware_perception_msgs::DynamicObjectArray & object);

geometry_msgs::Point toRosPoint(const pcl::PointXYZ & pcl_point);
geometry_msgs::Point toRosPoint(const Point2d & boost_point, const double z);

template <class T>
std::vector<T> concatVector(const std::vector<T> & vec1, const std::vector<T> & vec2)
{
  auto concat_vec = vec1;
  concat_vec.insert(std::end(concat_vec), std::begin(vec2), std::end(vec2));
  return concat_vec;
}
}  // namespace planning_utils

#endif  // COMMON_MATH_PLANNING_UTILS_H
