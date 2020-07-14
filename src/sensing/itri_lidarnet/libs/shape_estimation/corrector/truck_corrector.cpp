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

#include "truck_corrector.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#define EIGEN_MPL2_ONLY
#include <Eigen/Core>
#include <Eigen/Geometry>

bool TruckCorrector::correct(CLUSTER_INFO& cluster_info)
{
  /*
    Eigen::Translation<double, 2> trans =
      Eigen::Translation<double, 2>(pose_output.position.x, pose_output.position.y);

    Eigen::Quaterniond quat(
      pose_output.orientation.w, pose_output.orientation.x, pose_output.orientation.y,
      pose_output.orientation.z);
    Eigen::Vector3d euler = quat.toRotationMatrix().eulerAngles(0, 1, 2);
    const double yaw = euler[2];
    Eigen::Rotation2Dd rotate(yaw);
  */
  Eigen::Translation<double, 2> trans =
      Eigen::Translation<double, 2>(cluster_info.obb_center.x, cluster_info.obb_center.y);
  const double yaw = cluster_info.obb_orient;
  Eigen::Rotation2Dd rotate(yaw);

  Eigen::Affine2d affine_mat;
  affine_mat = trans * rotate.toRotationMatrix();

  /*
   *         ^ x
   *         |
   *        (0)
   * y       |
   * <--(1)--|--(3)--
   *         |
   *        (2)
   *         |
   */
  std::vector<Eigen::Vector2d> v_point;
  /*
    v_point.push_back(Eigen::Vector2d(shape_output.dimensions.x / 2.0, 0.0));
    v_point.push_back(Eigen::Vector2d(0.0, shape_output.dimensions.y / 2.0));
    v_point.push_back(Eigen::Vector2d(-shape_output.dimensions.x / 2.0, 0.0));
    v_point.push_back(Eigen::Vector2d(0.0, -shape_output.dimensions.y / 2.0));
  */
  v_point.push_back(Eigen::Vector2d(cluster_info.obb_dx / 2.0, 0.0));
  v_point.push_back(Eigen::Vector2d(0.0, cluster_info.obb_dy / 2.0));
  v_point.push_back(Eigen::Vector2d(-cluster_info.obb_dx / 2.0, 0.0));
  v_point.push_back(Eigen::Vector2d(0.0, -cluster_info.obb_dy / 2.0));

  size_t first_most_distant_index;
  {
    double distance = 0.0;
    for (size_t i = 0; i < v_point.size(); ++i)
    {
      if (distance < (affine_mat * v_point.at(i)).norm())
      {
        distance = (affine_mat * v_point.at(i)).norm();
        first_most_distant_index = i;
      }
    }
  }
  size_t second_most_distant_index;
  {
    double distance = 0.0;
    for (size_t i = 0; i < v_point.size(); ++i)
    {
      if (distance < (affine_mat * v_point.at(i)).norm() && i != first_most_distant_index)
      {
        distance = (affine_mat * v_point.at(i)).norm();
        second_most_distant_index = i;
      }
    }
  }
  size_t third_most_distant_index;
  {
    double distance = 0.0;
    for (size_t i = 0; i < v_point.size(); ++i)
    {
      if ((distance < (affine_mat * v_point.at(i)).norm()) && i != first_most_distant_index &&
          i != second_most_distant_index)
      {
        distance = (affine_mat * v_point.at(i)).norm();
        third_most_distant_index = i;
      }
    }
  }

  // rule based correction
  Eigen::Vector2d correction_vector = Eigen::Vector2d::Zero();
  constexpr double min_width = 1.5;
  constexpr double max_width = 2.9;
  constexpr double min_length = 4.0;
  constexpr double max_length = 7.9;

  if ((int)std::abs((int)first_most_distant_index - (int)second_most_distant_index) % 2 == 0)
  {
    if (min_width < (v_point.at(first_most_distant_index) * 2.0).norm() &&
        (v_point.at(first_most_distant_index) * 2.0).norm() < max_width)
    {
      if ((v_point.at(third_most_distant_index) * 2.0).norm() < max_length)
      {
        correction_vector = v_point.at(third_most_distant_index);
        if (correction_vector.x() == 0.0)
          correction_vector.y() = std::max(std::abs(correction_vector.y()), ((min_length + max_length) / 2.0) / 2.0) *
                                      (correction_vector.y() < 0.0 ? -1.0 : 1.0) -
                                  correction_vector.y();
        else if (correction_vector.y() == 0.0)
          correction_vector.x() = std::max(std::abs(correction_vector.x()), ((min_length + max_length) / 2.0) / 2.0) *
                                      (correction_vector.x() < 0.0 ? -1.0 : 1.0) -
                                  correction_vector.x();
        else
          return false;
      }
      else
      {
        return false;
      }
    }
    else if (min_length < (v_point.at(first_most_distant_index) * 2.0).norm() &&
             (v_point.at(first_most_distant_index) * 2.0).norm() < max_length)
    {
      if ((v_point.at(third_most_distant_index) * 2.0).norm() < max_width)
      {
        correction_vector = v_point.at(third_most_distant_index);
        if (correction_vector.x() == 0.0)
          correction_vector.y() = std::max(std::abs(correction_vector.y()), ((min_width + max_width) / 2.0) / 2.0) *
                                      (correction_vector.y() < 0.0 ? -1.0 : 1.0) -
                                  correction_vector.y();
        else if (correction_vector.y() == 0.0)
          correction_vector.x() = std::max(std::abs(correction_vector.x()), ((min_width + max_width) / 2.0) / 2.0) *
                                      (correction_vector.x() < 0.0 ? -1.0 : 1.0) -
                                  correction_vector.x();
        else
          return false;
      }
      else
      {
        return false;
      }
    }
    else
    {
      return false;
    }
  }
  // fit width
  else if ((min_width < (v_point.at(first_most_distant_index) * 2.0).norm() &&
            (v_point.at(first_most_distant_index) * 2.0).norm() < max_width) &&
           (min_width < (v_point.at(second_most_distant_index) * 2.0).norm() &&
            (v_point.at(second_most_distant_index) * 2.0).norm() < max_width))
  {
    correction_vector = v_point.at(first_most_distant_index);
    if (correction_vector.x() == 0.0)
      correction_vector.y() = std::max(std::abs(correction_vector.y()), ((min_length + max_length) / 2.0) / 2.0) *
                                  (correction_vector.y() < 0.0 ? -1.0 : 1.0) -
                              correction_vector.y();
    else if (correction_vector.y() == 0.0)
      correction_vector.x() = std::max(std::abs(correction_vector.x()), ((min_length + max_length) / 2.0) / 2.0) *
                                  (correction_vector.x() < 0.0 ? -1.0 : 1.0) -
                              correction_vector.x();
    else
      return false;
  }
  else if (min_width < (v_point.at(first_most_distant_index) * 2.0).norm() &&
           (v_point.at(first_most_distant_index) * 2.0).norm() < max_width)
  {
    correction_vector = v_point.at(second_most_distant_index);
    if (correction_vector.x() == 0.0)
      correction_vector.y() = std::max(std::abs(correction_vector.y()), ((min_length + max_length) / 2.0) / 2.0) *
                                  (correction_vector.y() < 0.0 ? -1.0 : 1.0) -
                              correction_vector.y();
    else if (correction_vector.y() == 0.0)
      correction_vector.x() = std::max(std::abs(correction_vector.x()), ((min_length + max_length) / 2.0) / 2.0) *
                                  (correction_vector.x() < 0.0 ? -1.0 : 1.0) -
                              correction_vector.x();
    else
      return false;
  }
  else if (min_width < (v_point.at(second_most_distant_index) * 2.0).norm() &&
           (v_point.at(second_most_distant_index) * 2.0).norm() < max_width)
  {
    correction_vector = v_point.at(first_most_distant_index);

    if (correction_vector.x() == 0.0)
      correction_vector.y() = std::max(std::abs(correction_vector.y()), ((min_length + max_length) / 2.0) / 2.0) *
                                  (correction_vector.y() < 0.0 ? -1.0 : 1.0) -
                              correction_vector.y();
    else if (correction_vector.y() == 0.0)
      correction_vector.x() = std::max(std::abs(correction_vector.x()), ((min_length + max_length) / 2.0) / 2.0) *
                                  (correction_vector.x() < 0.0 ? -1.0 : 1.0) -
                              correction_vector.x();
    else
      return false;
  }
  // fit length
  else if ((min_length < (v_point.at(first_most_distant_index) * 2.0).norm() &&
            (v_point.at(first_most_distant_index) * 2.0).norm() < max_length) &&
           (v_point.at(second_most_distant_index) * 2.0).norm() < max_width)
  {
    correction_vector = v_point.at(second_most_distant_index);

    if (correction_vector.x() == 0.0)
      correction_vector.y() = std::max(std::abs(correction_vector.y()), ((min_width + max_width) / 2.0) / 2.0) *
                                  (correction_vector.y() < 0.0 ? -1.0 : 1.0) -
                              correction_vector.y();
    else if (correction_vector.y() == 0.0)
      correction_vector.x() = std::max(std::abs(correction_vector.x()), ((min_width + max_width) / 2.0) / 2.0) *
                                  (correction_vector.x() < 0.0 ? -1.0 : 1.0) -
                              correction_vector.x();
    else
      return false;
  }
  else if ((min_length < (v_point.at(second_most_distant_index) * 2.0).norm() &&
            (v_point.at(second_most_distant_index) * 2.0).norm() < max_length) &&
           (v_point.at(first_most_distant_index) * 2.0).norm() < max_width)
  {
    correction_vector = v_point.at(first_most_distant_index);

    if (correction_vector.x() == 0.0)
      correction_vector.y() = std::max(std::abs(correction_vector.y()), ((min_width + max_width) / 2.0) / 2.0) *
                                  (correction_vector.y() < 0.0 ? -1.0 : 1.0) -
                              correction_vector.y();
    else if (correction_vector.y() == 0.0)
      correction_vector.x() = std::max(std::abs(correction_vector.x()), ((min_width + max_width) / 2.0) / 2.0) *
                                  (correction_vector.x() < 0.0 ? -1.0 : 1.0) -
                              correction_vector.x();
    else
      return false;
  }
  else
  {
    return false;
  }

  cluster_info.obb_dx += std::abs(correction_vector.x()) * 2.0;
  cluster_info.obb_dy += std::abs(correction_vector.y()) * 2.0;
  cluster_info.obb_center.x += (rotate.toRotationMatrix() * correction_vector).x();
  cluster_info.obb_center.y += (rotate.toRotationMatrix() * correction_vector).y();

  // correct to set long length is x, short length is y
  if (cluster_info.obb_dx < cluster_info.obb_dy)
  {
    double yaw_90_rotated = yaw + M_PI_2;
    cluster_info.obb_orient = yaw_90_rotated;
    double temp = cluster_info.obb_dx;
    cluster_info.obb_dx = cluster_info.obb_dy;
    cluster_info.obb_dy = temp;
  }

  cluster_info.obb_vertex.resize(8);
  double dxy[2], p0[2], p1[2], p2[2], p3[2];
  double xy[2] = { cluster_info.obb_center.x, cluster_info.obb_center.y };
  rotation2D(xy, dxy, cluster_info.obb_orient);
  double dp0[2] = { dxy[0] - (cluster_info.obb_dx / 2), dxy[1] - (cluster_info.obb_dy / 2) };
  double dp1[2] = { dxy[0] + (cluster_info.obb_dx / 2), dxy[1] - (cluster_info.obb_dy / 2) };
  double dp2[2] = { dxy[0] + (cluster_info.obb_dx / 2), dxy[1] + (cluster_info.obb_dy / 2) };
  double dp3[2] = { dxy[0] - (cluster_info.obb_dx / 2), dxy[1] + (cluster_info.obb_dy / 2) };
  rotation2D(dp0, p0, -cluster_info.obb_orient);
  rotation2D(dp1, p1, -cluster_info.obb_orient);
  rotation2D(dp2, p2, -cluster_info.obb_orient);
  rotation2D(dp3, p3, -cluster_info.obb_orient);
  setBoundingBox(cluster_info, p0, p1, p2, p3);

  /*
    shape_output.dimensions.x += std::abs(correction_vector.x()) * 2.0;
    shape_output.dimensions.y += std::abs(correction_vector.y()) * 2.0;
    pose_output.position.x += (rotate.toRotationMatrix() * correction_vector).x();
    pose_output.position.y += (rotate.toRotationMatrix() * correction_vector).y();

    // correct to set long length is x, short length is y
    if (shape_output.dimensions.x < shape_output.dimensions.y) {
      double roll, pitch, yaw;
      tf2::Quaternion quaternion;
      tf2::fromMsg(pose_output.orientation, quaternion);
      tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);
      double yaw_90_rotated = yaw + M_PI_2;
      tf2::Quaternion corrected_quaternion;
      corrected_quaternion.setRPY(roll, pitch, yaw_90_rotated);
      pose_output.orientation = tf2::toMsg(corrected_quaternion);
      double temp = shape_output.dimensions.x;
      shape_output.dimensions.x = shape_output.dimensions.y;
      shape_output.dimensions.y = temp;
    }
  */

  return true;
}
