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
#include "filter/car_filter.hpp"
#include "model/bounding_box.hpp"
#include "model/model_interface.hpp"

ShapeEstimator::ShapeEstimator()
{
}

bool ShapeEstimator::getShapeAndPose(CLUSTER_INFO& cluster_info)
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
  if (!applyFilter(cluster_info))
  {
    return false;
  }

  // rule based corrector
  if (!applyCorrector(cluster_info))
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

bool ShapeEstimator::applyFilter(const CLUSTER_INFO& cluster_info)
{
  // rule based filter
  std::unique_ptr<ShapeEstimationFilterInterface> filter_ptr;
  filter_ptr.reset(new CarFilter);
  return filter_ptr->filter(cluster_info);
}

bool ShapeEstimator::applyCorrector(CLUSTER_INFO& cluster_info)
{
  // rule based corrector
  std::unique_ptr<ShapeEstimationCorrectorInterface> corrector_ptr;
  corrector_ptr.reset(new CarCorrector);
  return corrector_ptr->correct(cluster_info);
}
