#ifndef __ALIGNMENT_JSON_READER_H__
#define __ALIGNMENT_JSON_READER_H__

#include <string>
#include "opencv2/core/mat.hpp"

namespace alignment
{
int read_distance_from_json(const std::string& filename, cv::Point3d** dist_in_cm, const int rows, const int cols);
};      // namespace
#endif  // __ALIGNMENT_JSON_READER_H__
