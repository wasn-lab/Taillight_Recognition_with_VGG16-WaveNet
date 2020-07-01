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

#pragma once

#include "corrector_interface.hpp"

class CarCorrector : public ShapeEstimationCorrectorInterface
{
private:
  void rotation2D(double xy_input[2], double xy_output[2], double theta);
  void setBoundingBox(CLUSTER_INFO & cluster_info, double pt0[2], double pt3[2], double pt4[2], double pt7[2]);

public:
  CarCorrector(){};

  ~CarCorrector(){};

  bool correct(CLUSTER_INFO & cluster_info) override;
};
