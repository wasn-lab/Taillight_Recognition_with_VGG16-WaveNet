/*
   CREATER: ICL U300
   DATE: July, 2019
 */

#include <memory>
#include <map>
#include <utility>
#include <fstream>
#include "pcl_ros/point_cloud.h"
#include "parknet_alignment_node.h"
#include "parknet_logging.h"
#include "alignment_utils.h"
#include "alignment_args_parser.h"
#include "alignment_json_writer.h"

ParknetAlignmentNode::ParknetAlignmentNode()
  : ParknetAlignmentNode(alignment::camera_image_width, alignment::camera_image_height)
{
}

ParknetAlignmentNode::ParknetAlignmentNode(const int width, const int height)
  : image_width_(width), image_height_(height)
{
  spatial_points_ = new cv::Point3d*[image_height_];
  assert(spatial_points_);
  num_pcd_received_ = 0;
  for (int i = 0; i < image_height_; i++)
  {
    spatial_points_[i] = new cv::Point3d[image_width_];
  }
  assert(spatial_points_[image_height_ - 1]);

  for (int row = 0; row < image_height_; row++)
  {
    for (int col = 0; col < image_width_; col++)
    {
      spatial_points_[row][col].x = INIT_COORDINATE_VALUE;
      spatial_points_[row][col].y = INIT_COORDINATE_VALUE;
      spatial_points_[row][col].z = INIT_COORDINATE_VALUE;
    }
  }
}

ParknetAlignmentNode::~ParknetAlignmentNode()
{
  LOG_INFO << "Generate distance mapping by " << num_pcd_received_ << " point cloud messages.";
  for (int row = 0; row < image_height_; row++)
  {
    delete[] spatial_points_[row];
  }
  delete[] spatial_points_;
}

bool ParknetAlignmentNode::is_valid_image_point(const int row, const int col) const
{
  bool ret = true;
  if ((col < 0) || (col >= image_width_) || (row < 0) || (row >= image_height_))
  {
    ret = false;
  }
  return ret;
}

bool ParknetAlignmentNode::spatial_point_is_valid(const cv::Point3d& point) const
{
  return (point.x != INIT_COORDINATE_VALUE) || (point.y != INIT_COORDINATE_VALUE) || (point.z != INIT_COORDINATE_VALUE);
}

bool ParknetAlignmentNode::spatial_point_is_valid(const int row, const int col) const
{
  if (is_valid_image_point(row, col))
  {
    return spatial_point_is_valid(spatial_points_[row][col]);
  }
  else
  {
    return false;
  }
}

int ParknetAlignmentNode::set_spatial_point(const int row, const int col, const cv::Point3d& in_point)
{
  assert(row >= 0 && row < image_height_);
  assert(col >= 0 && col < image_width_);
  spatial_points_[row][col] = in_point;
  return 0;
}

cv::Point3d ParknetAlignmentNode::get_spatial_point(const int row, const int col) const
{
  assert(row >= 0 && row < image_height_);
  assert(col >= 0 && col < image_width_);
  return spatial_points_[row][col];
}

bool ParknetAlignmentNode::search_valid_neighbor(const int row, const int col, cv::Point* valid_neighbor) const
{
  for (int roffset = -1; roffset <= 1; roffset++)
  {
    for (int coffset = -1; coffset <= 1; coffset++)
    {
      if ((roffset == 0) and (coffset == 0))
      {
        continue;
      }
      int nearby_row = row + roffset;
      int nearby_col = col + coffset;
      if (spatial_point_is_valid(nearby_row, nearby_col))
      {
        valid_neighbor->x = nearby_row;
        valid_neighbor->y = nearby_col;
        VLOG(2) << "(" << row << ", " << col << ") -> (" << nearby_row << ", " << nearby_col
                << ") with spatial coordinate:" << spatial_points_[nearby_row][nearby_col].x << " "
                << spatial_points_[nearby_row][nearby_col].y << " " << spatial_points_[nearby_row][nearby_col].z;
        return true;
      }
    }
  }
  return false;
}

void ParknetAlignmentNode::pcd_callback(const boost::shared_ptr<const sensor_msgs::PointCloud2>& in_cloud)
{
  assert(in_cloud->width > 0);

  pcl::fromROSMsg(*in_cloud, pcd_);

  const int cam_sn = alignment_args_parser::get_cam_sn();
  LOG_INFO << "pcd_.size(): " << pcd_.size();
  num_pcd_received_ += 1;
  for (size_t i = 0; i < pcd_.size(); i++)
  {
    const auto image_point = alignment::map_pcd_point_to_image_point(pcd_.points[i], cam_sn);
    const auto x = image_point.x;
    const auto y = image_point.y;
    if (is_valid_image_point(x, y))
    {
      spatial_points_[y][x].x = pcd_.points[i].x;
      spatial_points_[y][x].y = pcd_.points[i].y;
      spatial_points_[y][x].z = pcd_.points[i].z;
    }
  }
}

void ParknetAlignmentNode::subscribe_and_advertise_topics()
{
  const int cam_sn = alignment_args_parser::get_cam_sn();
  pcd_subscribers_ = node_handle_.subscribe<sensor_msgs::PointCloud2>(alignment::get_lidar_topic_by_cam_sn(cam_sn), 2,
                                                                      &ParknetAlignmentNode::pcd_callback, this);
}

void ParknetAlignmentNode::approx_nearest_points_if_necessary()
{
  LOG_INFO << __FUNCTION__;
  std::vector<cv::Point> unset_points;
  bool done = false;

  for (int row = 0; row < image_height_; row++)
  {
    for (int col = 0; col < image_width_; col++)
    {
      if (!spatial_point_is_valid(row, col))
      {
        unset_points.emplace_back(cv::Point(row, col));
      }
    }
  }
  LOG_INFO << "Total " << unset_points.size() << " need to be approximated";
  while (!done)
  {
    int num_approx = 0;
    std::vector<cv::Point> unset_points_temp;
    std::map<std::pair<int, int>, std::pair<int, int>> revised_points;
    unset_points_temp.reserve(unset_points.size());
    for (const auto& point : unset_points)
    {
      auto img_row = point.x;
      auto img_col = point.y;
      cv::Point valid_nearby_point;
      if (search_valid_neighbor(img_row, img_col, &valid_nearby_point))
      {
        std::pair<int, int> image_point, aligned_image_point;
        auto nearby_row = valid_nearby_point.x;
        auto nearby_col = valid_nearby_point.y;
        image_point = std::make_pair(img_row, img_col);
        aligned_image_point = std::make_pair(nearby_row, nearby_col);

        revised_points[image_point] = aligned_image_point;
        num_approx++;
      }
      else
      {
        unset_points_temp.emplace_back(cv::Point(img_row, img_col));
      }
    }

    for (const auto& kv : revised_points)
    {
      const auto& image_point = kv.first;
      const auto& aligned_image_point = kv.second;
      spatial_points_[image_point.first][image_point.second] =
          spatial_points_[aligned_image_point.first][aligned_image_point.second];
    }

    LOG_INFO << "Total " << unset_points_temp.size() << " need to be approximated";
    unset_points.assign(unset_points_temp.begin(), unset_points_temp.end());
    if ((unset_points.size() == 0) || (num_approx == 0))
    {
      done = true;
    }
  }
  LOG_INFO << " unset_points: " << unset_points.size();
}

void ParknetAlignmentNode::dump_dist_mapping() const
{
  LOG_INFO << __FUNCTION__;

  std::unique_ptr<std::unique_ptr<double[]>[]> dist(new std::unique_ptr<double[]>[image_height_]);
  for (int i = 0; i < image_height_; i++)
  {
    dist[i] = std::unique_ptr<double[]>(new double[image_width_]);
  }

  cv::Mat grayscale(image_height_, image_width_, CV_8U);

  double max_dist = 0.01;
  for (int i = 0; i < image_height_; i++)
  {
    for (int j = 0; j < image_width_; j++)
    {
      const auto x = spatial_points_[i][j].x;
      const auto y = spatial_points_[i][j].y;
      dist[i][j] = sqrt(x * x + y * y);
      if (dist[i][j] > max_dist)
      {
        max_dist = dist[i][j];
      }
    }
  }
  LOG_INFO << "max_dist: " << max_dist << " meters.";

  for (int i = 0; i < image_height_; i++)
  {
    for (int j = 0; j < image_width_; j++)
    {
      grayscale.at<uchar>(i, j) = (dist[i][j] * 255 / max_dist);
      //      LOG_INFO << "( " << i << ", " << j << ") grayscale: " << grayscale.at<int>(i, j) << " dist:" <<
      //      dist[i][j];
    }
  }
  cv::imwrite("dist_grayscale.ppm", grayscale);
  LOG_INFO << "Write dist_grayscale.ppm";
}

void ParknetAlignmentNode::dump_distance_in_json() const
{
  auto json_string = alignment::jsonize_spatial_points(spatial_points_, image_height_, image_width_);
  std::string filename = alignment_args_parser::get_output_filename();
  std::ofstream ofs(filename);
  ofs << json_string;
  LOG_INFO << "Write to " << filename;
}

void ParknetAlignmentNode::run(int argc, char* argv[])
{
  subscribe_and_advertise_topics();
  ros::Rate r(10);
  while (ros::ok() && (num_pcd_received_ < alignment_args_parser::get_pcd_nums()))
  {
    ros::spinOnce();
    r.sleep();
  }
  approx_nearest_points_if_necessary();
  dump_dist_mapping();
  dump_distance_in_json();
}
