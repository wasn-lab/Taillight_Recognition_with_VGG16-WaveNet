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

#include "shape_estimator.hpp"
#include <iostream>
#include <memory>
#include "corrector/car_corrector.hpp"
#include "corrector/truck_corrector.hpp"
#include "corrector/bus_corrector.hpp"
#include "corrector/motor_corrector.hpp"
#include "corrector/pedestrian_corrector.hpp"
#include "corrector/no_corrector.hpp"
#include "filter/car_filter.hpp"
#include "filter/truck_filter.hpp"
#include "filter/bus_filter.hpp"
#include "filter/motor_filter.hpp"
#include "filter/pedestrian_filter.hpp"
#include "filter/no_filter.hpp"
#include "model/bounding_box.hpp"

ShapeEstimator::ShapeEstimator()
{
}

bool ShapeEstimator::getShapeAndPose(nnClassID type, CLUSTER_INFO& cluster_info, bool do_apply_filter)
{
  // check input
  if (cluster_info.cloud.empty())
  {
    return false;
  }

  // estimate shape
  if (!estimateShape(cluster_info))
  {
    return false;
  }

  if (do_apply_filter)
  {
    // rule based filter
    if (!applyFilter(type, cluster_info))
    {
      return false;
    }

    // rule based corrector
    if (!applyCorrector(type, cluster_info))
    {
      return false;
    }
  }
  else
  {
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
  }

  return true;
}

bool ShapeEstimator::getShapeAndPose(nnClassID type, CLUSTER_INFO& cluster_info)
{
  // check input
  if (cluster_info.cloud.empty())
    return false;

  // estimate shape
  if (!estimateShape(cluster_info))
  {
    return false;
  }

  // rule based filter
  if (!applyFilter(type, cluster_info))
  {
    return false;
  }

  // rule based corrector
  if (!applyCorrector(type, cluster_info))
  {
    return false;
  }
  return true;
}

bool ShapeEstimator::estimateShape(CLUSTER_INFO& cluster_info)
{
  // estimate shape
  std::unique_ptr<ShapeEstimationModelInterface> model_ptr;
  model_ptr.reset(new BoundingBoxModel);
  return model_ptr->estimate(cluster_info);
}

bool ShapeEstimator::applyFilter(const nnClassID type, const CLUSTER_INFO& cluster_info)
{
  // rule based filter
  std::unique_ptr<ShapeEstimationFilterInterface> filter_ptr;
  if (type == nnClassID::Car && (cluster_info.obb_dx < 5.0 && cluster_info.obb_dy < 5.0))
  {
    filter_ptr.reset(new CarFilter);
  }
  else if (type == nnClassID::Car && (cluster_info.obb_dx < 7.9 && cluster_info.obb_dy < 7.9))
  {
    filter_ptr.reset(new TruckFilter);
  }
  else if (type == nnClassID::Car && (cluster_info.obb_dx < 12 && cluster_info.obb_dy < 12))
  {
    filter_ptr.reset(new BusFilter);
  }
  else if (type == nnClassID::Motobike)
  {
    filter_ptr.reset(new MotorFilter);
  }
  else if (type == nnClassID::Person)
  {
    filter_ptr.reset(new PedestrianFilter);
  }
  else
  {
    filter_ptr.reset(new NoFilter);
  }
  return filter_ptr->filter(cluster_info);
}

bool ShapeEstimator::applyCorrector(const nnClassID type, CLUSTER_INFO& cluster_info)
{
  // rule based corrector
  std::unique_ptr<ShapeEstimationCorrectorInterface> corrector_ptr;
  if (type == nnClassID::Car && (cluster_info.obb_dx < 5.0 && cluster_info.obb_dy < 5.0))
  {
    corrector_ptr.reset(new CarCorrector);
  }
  else if (type == nnClassID::Car && (cluster_info.obb_dx < 7.9 && cluster_info.obb_dy < 7.9))
  {
    corrector_ptr.reset(new TruckCorrector);
  }
  else if (type == nnClassID::Car && (cluster_info.obb_dx < 12 && cluster_info.obb_dy < 12))
  {
    corrector_ptr.reset(new BusCorrector);
  }
  else if (type == nnClassID::Motobike)
  {
    corrector_ptr.reset(new MotorCorrector);
  }
  else if (type == nnClassID::Person)
  {
    corrector_ptr.reset(new PedestrianCorrector);
  }
  else
  {
    corrector_ptr.reset(new NoCorrector);
  }
  return corrector_ptr->correct(cluster_info);
}

void ShapeEstimator::rotation2D(double xy_input[2], double xy_output[2], double theta)
{
  xy_output[0] = xy_input[0] * std::cos(theta) + xy_input[1] * std::sin(theta);
  xy_output[1] = -xy_input[0] * std::sin(theta) + xy_input[1] * std::cos(theta);
}

void ShapeEstimator::setBoundingBox(CLUSTER_INFO& cluster_info, double pt0[2], double pt3[2], double pt4[2], double pt7[2])
{
  /* ABB order
  | pt5 _______pt6(max)
  |     |\      \
  |     | \      \
  |     |  \______\
  | pt4 \  |pt1   |pt2
  |      \ |      |
  |       \|______|
  | pt0(min)    pt3
  |------------------------>X
  */
  double z_min = cluster_info.obb_center.z - (cluster_info.obb_dz / 2);
  double z_max = cluster_info.obb_center.z + (cluster_info.obb_dz / 2);
  cluster_info.obb_vertex.at(1) = pcl::PointXYZ(pt0[0], pt0[1], z_max);
  cluster_info.obb_vertex.at(2) = pcl::PointXYZ(pt3[0], pt3[1], z_max);
  cluster_info.obb_vertex.at(6) = pcl::PointXYZ(pt4[0], pt4[1], z_max);
  cluster_info.obb_vertex.at(5) = pcl::PointXYZ(pt7[0], pt7[1], z_max);
  cluster_info.obb_vertex.at(0) = pcl::PointXYZ(pt0[0], pt0[1], z_min);
  cluster_info.obb_vertex.at(3) = pcl::PointXYZ(pt3[0], pt3[1], z_min);
  cluster_info.obb_vertex.at(7) = pcl::PointXYZ(pt4[0], pt4[1], z_min);
  cluster_info.obb_vertex.at(4) = pcl::PointXYZ(pt7[0], pt7[1], z_min);
}