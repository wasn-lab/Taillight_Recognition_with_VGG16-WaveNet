#include <cstdio>

class SkeletonBuffer
{
public:
  ros::Time timestamp;
  int track_id;
  std::vector<cv::Point2f> last_real_skeleton;
  std::vector<std::vector<cv::Point2f>> calculated_skeleton;
};
