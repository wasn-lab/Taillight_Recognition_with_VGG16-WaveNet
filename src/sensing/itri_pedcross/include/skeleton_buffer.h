#include <cstdio>

class SkeletonBuffer
{
public:
  ros::Time timestamp_;
  unsigned int track_id_;
  // calculated_skeleton is for storing skip frame predicted keypoints
  std::vector<std::vector<cv::Point2f>> calculated_skeleton_;
  // stored_skeleton is for storing 10 frames keypoints for cross prediction
  std::vector<std::vector<cv::Point2f>> stored_skeleton_;
  // store bbox for recalculating features
  std::vector<std::vector<float>> data_bbox_;

  std::vector<double> history_distance_from_path_;
  std::vector<cv::Point2f> history_position_;

  cv::Mat image_for_optical_flow_;
};