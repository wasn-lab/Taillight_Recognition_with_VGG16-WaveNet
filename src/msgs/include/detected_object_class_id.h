#ifndef __DETECTED_OBJECT_CLASS_ID_H__
#define __DETECTED_OBJECT_CLASS_ID_H__

namespace sensor_msgs_itri
{
enum DetectedObjectClassId
{
  Unknown,
  Person,
  Bicycle,
  Motobike,
  Car,
  Bus,
  Truck,
  Sign,
  Light,
  NumerOfId
};

static_assert(DetectedObjectClassId::Unknown == 0, "");
static_assert(DetectedObjectClassId::Person == 1, "");
static_assert(DetectedObjectClassId::NumerOfId == 9, "");

}  // namespace sensor_msgs_itri

#endif  // __DETECTED_OBJECT_CLASS_ID_H__

