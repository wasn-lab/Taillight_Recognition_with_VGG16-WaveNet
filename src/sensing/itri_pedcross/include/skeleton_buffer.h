#include <cstdio>

class SkeletonBuffer
{
public:
  ros::Time timestamp;
  unsigned int track_id;
  // calculated_skeleton is for storing skip frame predicted keypoints
  std::vector<std::vector<cv::Point2f>> calculated_skeleton;
  // stored_skeleton is for storing 10 frames keypoints for cross prediction
  std::vector<std::vector<cv::Point2f>> stored_skeleton;
  // store bbox for recalculating features
  std::vector<std::vector<float>> data_bbox;

  std::vector<double> history_distance_from_path;
  std::vector<cv::Point2f> history_position;

  cv::Mat image_for_optical_flow;
};