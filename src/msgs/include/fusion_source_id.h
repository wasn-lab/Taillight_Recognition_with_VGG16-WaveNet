#ifndef __FUSION_SOURCE_ID_H__
#define __FUSION_SOURCE_ID_H__

namespace sensor_msgs_itri
{
enum FusionSourceId
{
  Camera,
  Radar,
  Lidar
};

static_assert(FusionSourceId::Camera == 0, "cam");
static_assert(FusionSourceId::Radar == 1, "rad");
static_assert(FusionSourceId::Lidar == 2, "lid");

}  // namespace sensor_msgs_itri

#endif  // __FUSION_SOURCE_ID_H__
