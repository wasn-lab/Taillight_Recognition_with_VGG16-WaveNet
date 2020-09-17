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

#include "bus_filter.hpp"

bool BusFilter::filter(const CLUSTER_INFO& cluster_info)
{
  double x = cluster_info.obb_dx;
  double y = cluster_info.obb_dy;
  double s = x * y;
  constexpr double min_width = 2.0;
  constexpr double max_width = 2.9;
  //constexpr double min_length = 5.0;
  constexpr double max_length = 12.0;

  if (x < min_width && y < min_width)
  {
    return false;
  }
  if (max_width < x && max_width < y)
  {
    return false;
  }

  if (max_length < x || max_length < y)
  {
    return false;
  }

  if (s < 0.5 && max_length * max_width < s)
  {
    return false;
  }
  return true;
}
