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

bool ShapeEstimator::getShapeAndPose(nnClassID type, CLUSTER_INFO & cluster_info)
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
  if (!applyFilter(type, cluster_info)) {
    return false;
  }

  // rule based corrector
  if (!applyCorrector(type, cluster_info)) {
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

bool ShapeEstimator::applyFilter(const nnClassID type, const CLUSTER_INFO & cluster_info)
{
  // rule based filter
  std::unique_ptr<ShapeEstimationFilterInterface> filter_ptr;
  if (type == nnClassID::Car && (cluster_info.obb_dx < 5.0 && cluster_info.obb_dy < 5.0)) {
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

bool ShapeEstimator::applyCorrector(const nnClassID type, CLUSTER_INFO & cluster_info)
{
  // rule based corrector
  std::unique_ptr<ShapeEstimationCorrectorInterface> corrector_ptr;
  if (type == nnClassID::Car && (cluster_info.obb_dx < 5.0 && cluster_info.obb_dy < 5.0)) {
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
