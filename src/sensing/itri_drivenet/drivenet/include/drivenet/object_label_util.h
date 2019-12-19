#ifndef OBJECTLABELUTIL_H_
#define OBJECTLABELUTIL_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace DriveNet
{
enum class net_type_id
{
  begin = 0,
  person = begin,  // 0
  bicycle,         // 1
  car,             // 2
  motorbike,       // 3
  others1,         // 4
  bus,             // 5
  others2,         // 6
  truck            // 7
};

enum class common_type_id
{
  begin = 0,
  other = begin,  // 0
  person,         // 1
  bicycle,        // 2
  motorbike,      // 3
  car,            // 4
  bus,            // 5
  truck           // 6
};

int translate_label(int label);
cv::Scalar get_labelColor(std::vector<cv::Scalar> colors, int label_id);
cv::Scalar get_commonLabelColor(std::vector<cv::Scalar> colors, int label_id);
}
#endif /*OBJECTLABELUTIL_H_*/