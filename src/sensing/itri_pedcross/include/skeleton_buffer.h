#include <cstdio>

class SkeletonBuffer
{
public:
  ros::Time timestamp;
  unsigned int track_id;
  std::vector<cv::Point2f> previous_two_real_skeleton;
  std::vector<cv::Point2f> previous_one_real_skeleton;
  std::vector<std::vector<cv::Point2f>> calculated_skeleton;
  std::vector<std::vector<cv::Point2f>> back_calculated_skeleton;
  std::vector<std::vector<cv::Point2f>> stored_skeleton;

  std::vector<double> history_distance_from_path;
};