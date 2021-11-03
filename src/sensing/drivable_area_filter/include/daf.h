#ifndef __DAF_H__
#define __DAF_H__

#include "daf_base.h"

namespace daf
{
enum InputSource
{
  Fusion = 0,
  LidarDet,
  LidarDet_PointPillars_Car,
  LidarDet_PointPillars_Ped_Cyc,
  VirtualBBoxAbs,
  VirtualBBoxRel,
  CameraDetV2,
  Tracking2D,
  NumInputSources
};
}  // namespace daf

#endif  // __DAF_H__
