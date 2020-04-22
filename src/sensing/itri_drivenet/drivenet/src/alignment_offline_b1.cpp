#include "drivenet/alignment_offline_b1.h"

AlignmentOff::AlignmentOff()
{
/// camera layout
#if CAR_MODEL_IS_B1_V2
  const camera::id camId = camera::id::front_bottom_60;
#elif CAR_MODEL_IS_B1
  const camera::id camId = camera::id::front_60;
#else
#error "car model is not well defined"
#endif

  pj.init(camId);

  spatial_points_ = new cv::Point3d*[imgW];
  assert(spatial_points_);
  for (int i = 0; i < imgH; i++)
  {
    spatial_points_[i] = new cv::Point3d[imgW];
  }
  assert(spatial_points_[imgH - 1]);

  for (int row = 0; row < imgH; row++)
  {
    for (int col = 0; col < imgW; col++)
    {
      spatial_points_[row][col].x = INIT_COORDINATE_VALUE;
      spatial_points_[row][col].y = INIT_COORDINATE_VALUE;
      spatial_points_[row][col].z = INIT_COORDINATE_VALUE;
    }
  }
}

AlignmentOff::~AlignmentOff()
{
  for (int row = 0; row < imgH; row++)
  {
    delete[] spatial_points_[row];
  }
  delete[] spatial_points_;
}

bool AlignmentOff::is_valid_image_point(const int row, const int col) const
{
  bool ret = true;
  if ((col < 0) || (col >= imgW) || (row < 0) || (row >= imgH))
  {
    ret = false;
  }
  return ret;
}

bool AlignmentOff::spatial_point_is_valid(const cv::Point3d& point) const
{
  return (point.x != INIT_COORDINATE_VALUE) || (point.y != INIT_COORDINATE_VALUE) || (point.z != INIT_COORDINATE_VALUE);
}

bool AlignmentOff::spatial_point_is_valid(const int row, const int col) const
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

std::vector<int> AlignmentOff::run(float x, float y, float z)
{
  return pj.project(x, y, z);
}

bool AlignmentOff::search_valid_neighbor(const int row, const int col, cv::Point* valid_neighbor) const
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
        // VLOG(2) << "(" << row << ", " << col << ") -> (" << nearby_row << ", " << nearby_col
        //         << ") with spatial coordinate:" << spatial_points_[nearby_row][nearby_col].x << " "
        //         << spatial_points_[nearby_row][nearby_col].y << " " << spatial_points_[nearby_row][nearby_col].z;
        return true;
      }
    }
  }
  return false;
}

void AlignmentOff::approx_nearest_points_if_necessary()
{
  std::vector<cv::Point> unset_points;
  cv::Mat tmpa(imgH, imgW, CV_8UC3);
  // std::vector<cv::Point> dis_esti_table;

  bool done = false;

  std::cout << "Starting to create image" << std::endl;

  /*
    for(int i = 0; i < 5000; i++)
    {
      std::cout << "Process: " << i << "/5000" << std::endl ;

      for(int j = -3000; j < 3000; j++)
      {
        float i_float = (float)i/100;
        float j_float = (float)j/100;
        float tmpz = (float)i_float*0.020686-2.63239;
        // float tmpz = ((float)i_float-79)/30;
      // if(LidAll_cloudPtr->points[i].x > 0 && abs(LidAll_cloudPtr->points[i].z - (LidAll_cloudPtr->points[i].x -
    79)/30) < 0.1)      //
        out = run((float)i_float, (float)j_float, tmpz);

        if (out[0] > 0 && out[0] < imgW && out[1] > 0 && out[1] < imgH)
        {
          // std::cout << "2D: x = " << out[0] << ". y = " << out[1] ;
          // std::cout << ", 3D: x = " << i_float << ". y = " << j_float << ". z = " << tmpz << std::endl ;

          spatial_points_[out[1]][out[0]].x = i_float;
          spatial_points_[out[1]][out[0]].y = j_float;
          spatial_points_[out[1]][out[0]].z = 0.0;
        }
      }
    }
    */

  for (int row = 0; row < imgH; row++)
  {
    for (int col = 0; col < imgW; col++)
    {
      if (!spatial_point_is_valid(row, col))
      {
        unset_points.emplace_back(cv::Point(row, col));
        tmpa.at<cv::Vec3b>(row, col)[0] = 255;
        tmpa.at<cv::Vec3b>(row, col)[1] = 255;
        tmpa.at<cv::Vec3b>(row, col)[2] = 255;
      }
    }
  }

  cv::namedWindow("image", 1);
  cv::imshow("image", tmpa);
  cv::waitKey();

  std::cout << "Total " << unset_points.size() << " need to be approximated" << std::endl;

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

    std::cout << "Total " << unset_points_temp.size() << " need to be approximated" << std::endl;
    unset_points.assign(unset_points_temp.begin(), unset_points_temp.end());
    if ((unset_points.empty() == 0) || (num_approx == 0))
    {
      done = true;
    }
  }
  std::cout << " unset_points: " << unset_points.size();
}

void AlignmentOff::dump_distance_in_json() const
{
  auto json_string = jsonize_spatial_points(spatial_points_, imgH, imgW);
  std::string filename = "out.json";
  std::ofstream ofs(filename);
  ofs << json_string;
  std::cout << "Write to " << filename << std::endl;
}

std::string AlignmentOff::jsonize_spatial_points(cv::Point3d** spatial_points, int rows, int cols) const
{
  assert(spatial_points);
  assert(spatial_points[0]);
  Json::Value jspatial_points(Json::arrayValue);
  Json::FastWriter jwriter;
  for (int row = 0; row < rows; row++)
  {
    for (int col = 0; col < cols; col++)
    {
      Json::Value jvalue;
      Json::Value dist_in_cm(Json::arrayValue);
      jvalue["im_x"] = col;
      jvalue["im_y"] = row;

      // distance is measured in meters
      dist_in_cm[0] = int(spatial_points[row][col].x * 100);
      dist_in_cm[1] = int(spatial_points[row][col].y * 100);
      dist_in_cm[2] = int(spatial_points[row][col].z * 100);
      jvalue["dist_in_cm"] = dist_in_cm;

      jspatial_points.append(jvalue);
    }
  }
  jwriter.omitEndingLineFeed();
  return jwriter.write(jspatial_points);
}

// Main
AlignmentOff g_al;

void callback_LidarAll(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr LidAll_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*msg, *LidAll_cloudPtr);

  for (size_t i = 0; i < LidAll_cloudPtr->size(); i++)
  {

    if (LidAll_cloudPtr->points[i].z > g_al.groundLowBound && LidAll_cloudPtr->points[i].z < g_al.groundUpBound &&
       LidAll_cloudPtr->points[i].x > 0 )
    // if(LidAll_cloudPtr->points[i].x > 0 && abs(LidAll_cloudPtr->points[i].z - (2*LidAll_cloudPtr->points[i].x - 122)/45) < 0.1)
    // if(LidAll_cloudPtr->points[i].x > 0 && abs(LidAll_cloudPtr->points[i].z - (LidAll_cloudPtr->points[i].x - 79)/30) < 0.1)
    {
      g_al.out = g_al.run(LidAll_cloudPtr->points[i].x, LidAll_cloudPtr->points[i].y, LidAll_cloudPtr->points[i].z);
      if (g_al.out[0] > 0 && g_al.out[0] < g_al.imgW && g_al.out[1] > 0 && g_al.out[1] < g_al.imgH)
      {
        g_al.spatial_points_[g_al.out[1]][g_al.out[0]].x = LidAll_cloudPtr->points[i].x;
        g_al.spatial_points_[g_al.out[1]][g_al.out[0]].y = LidAll_cloudPtr->points[i].y;
        g_al.spatial_points_[g_al.out[1]][g_al.out[0]].z = LidAll_cloudPtr->points[i].z;

        // std::cout << LidAll_cloudPtr->points[i].x;
        // std::cout << LidAll_cloudPtr->points[i].y;
        // std::cout << LidAll_cloudPtr->points[i].z << std::endl;

        // std::cout << g_al.spatial_points_[g_al.out[1]][g_al.out[0]].x;
        // std::cout << g_al.spatial_points_[g_al.out[1]][g_al.out[0]].y;
        // std::cout << g_al.spatial_points_[g_al.out[1]][g_al.out[0]].z << std::endl;
      }
    }

    // if(LidAll_cloudPtr->points[i].x > 25 && LidAll_cloudPtr->points[i].z < 0 && LidAll_cloudPtr->points[i].y < 1 && LidAll_cloudPtr->points[i].y > -1)
    // {
    //   g_al.out = g_al.run(LidAll_cloudPtr->points[i].x, LidAll_cloudPtr->points[i].y, LidAll_cloudPtr->points[i].z);
    //   if(g_al.out[0] > 0 && g_al.out[0] < g_al.imgW && g_al.out[1] > 0 && g_al.out[1] < g_al.imgH)
    //   {
    //     if(g_al.out[0] = 304)
    //     {
    //       std::cout << LidAll_cloudPtr->points[i].x << ",";
    //       std::cout << LidAll_cloudPtr->points[i].y << ",";
    //       std::cout << LidAll_cloudPtr->points[i].z << ",";
    //       std::cout << g_al.out[0] << ",";
    //       std::cout << g_al.out[1] << std::endl;
    //     }
    //   }
    


    
  }
}

int main(int argc, char** argv)
{
  // new
  ros::init(argc, argv, "alignmentOff");
  ros::NodeHandle nh;
  ros::Rate r(10);

  ros::Subscriber LidarSc;
  LidarSc = nh.subscribe("LidarAll", 1, callback_LidarAll);

  while (ros::ok())
  {
    ros::spinOnce();
    r.sleep();
  }

  g_al.approx_nearest_points_if_necessary();
  g_al.dump_distance_in_json();
}