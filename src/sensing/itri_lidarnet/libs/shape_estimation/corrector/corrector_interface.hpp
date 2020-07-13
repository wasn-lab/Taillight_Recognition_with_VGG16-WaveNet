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
 * v1.0 Yukihiro Saito
 */

#pragma once

#include <string>
#include <cmath>
#include "../../UserDefine.h"

class ShapeEstimationCorrectorInterface
{
public:

  ShapeEstimationCorrectorInterface(){};

  virtual ~ShapeEstimationCorrectorInterface(){};

  virtual bool correct(CLUSTER_INFO & cluster_info) = 0;

  void rotation2D(double xy_input[2], double xy_output[2], double theta)
  {
    xy_output[0] = xy_input[0] * std::cos(theta) + xy_input[1] * std::sin(theta);
    xy_output[1] = -xy_input[0] * std::sin(theta) + xy_input[1] * std::cos(theta);
  }

  void setBoundingBox(CLUSTER_INFO& cluster_info, double pt0[2], double pt3[2], double pt4[2], double pt7[2])
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
};
