#ifndef __NETOUTPUT__
#define __NETOUTPUT__

#include <string>
#include <vector>

namespace SensingSubSystem
{
template <typename T>
class PerceptionOutput
{
public:
  std::vector<T> dobj;  // the variables of output objects, e.g. lane, sign, light

  void ToROSMsg();  // to be edited
};

// LaneNet definition
typedef enum {
  LANE_ADJACENT_LEFT = 0,
  LANE_EGO_LEFT = 1,
  LANE_EGO_RIGHT = 2,
  LANE_ADJACENT_RIGHT = 3,
  LANE_EGO_MID = 4
} laneType;

struct LaneNetObject
{
  laneType classID;
  float a;  // 3rd parameter
  float b;  // 2nd parameter
  float c;  // 1st parameter
  float d;  // constant
  float disStopLine;
};

// SignNet definition
typedef enum {
  SIGN_UNKNOWN = 0,
  SIGN_RAILROAD_CROSSING = 1,
  SIGN_TRAFFIC_SIGNAL = 2,
  SIGN_CLOSED_TO_VEHICLES = 3,
  SIGN_NO_ENTRY = 4,
  SIGN_NO_PARKING = 5,
  SIGN_LEFT_DIRECTION_ONLY = 6,
  SIGN_ONE_WAY = 7,
  SIGN_SPEED_LIMIT = 8,
  SIGN_SLOW = 9,
  SIGN_STOP = 10
} signType;

struct SignNetObject
{
  signType classID;
  float distance;
  unsigned content;  // additional info
};

// LightNet definition
typedef enum {
  SIGNAL_UNKNOWN = 0,
  SIGNAL_RED = 1,
  SIGNAL_YELLOW = 2,
  SIGNAL_GREEN = 3,
  SIGNAL_RED_RIGHT = 4,
  SIGNAL_PED_GO = 5,
  SIGNAL_PED_STOP = 6,
  SIGNAL_PED_BLINK_GO = 7
} signalType;

struct LightObject
{
  signalType classID;
  float distance;
  unsigned liType;
};
// DriveNet definition
// DriveNet definition
struct PointXYZ
{
  float x;
  float y;
  float z;
};
struct BoxPoint
{
  PointXYZ p0;
  PointXYZ p1;
  PointXYZ p2;
  PointXYZ p3;
  PointXYZ p4;
  PointXYZ p5;
  PointXYZ p6;
  PointXYZ p7;
};
struct DriveNetObject
{
  int classId;
  int u;
  int v;
  int width;
  int height;
  float prob;
  BoxPoint bPoint;
};
}

#endif
