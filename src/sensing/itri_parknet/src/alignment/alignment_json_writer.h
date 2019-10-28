#ifndef __ALIGNMENT_JSON_WRITER_H__
#define __ALIGNMENT_JSON_WRITER_H__

#include <string>
#include "opencv2/core/mat.hpp"

namespace alignment
{
std::string jsonize_spatial_points(cv::Point3d** spatial_points_, const int rows, const int cols);
};      // namespace
#endif  // __ALIGNMENT_JSON_WRITER_H__
