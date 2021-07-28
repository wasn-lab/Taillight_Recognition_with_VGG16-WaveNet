/**
 * @file
 * @brief PCL point datatype for use with the OS-1
 */

#pragma once
#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>

namespace ouster_ros
{
namespace OS1
{
struct EIGEN_ALIGN16 PointXYZIR
{
  PCL_ADD_POINT4D;
  float intensity;
  uint8_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static inline PointXYZIR make(float x, float y, float z, float intensity, uint8_t ring)
  {
    return { x, y, z, 0.0, intensity, ring };
  }
};
}  // namespace OS1
}  // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::OS1::PointXYZIR,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint8_t, ring, ring)
)
// clang-format on
