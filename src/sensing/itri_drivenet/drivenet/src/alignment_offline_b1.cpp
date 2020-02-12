#include "drivenet/alignment_offline_b1.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void AlignmentOff::init(int car_id)
{
  carId = car_id;
}

