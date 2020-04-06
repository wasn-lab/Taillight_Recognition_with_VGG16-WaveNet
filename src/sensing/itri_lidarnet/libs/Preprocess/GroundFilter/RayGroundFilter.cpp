#include "RayGroundFilter.h"

RayGroundFilter::RayGroundFilter(float sensor_height, float general_max_slope, float local_max_slope,
                                 float radial_divider_angle, float concentric_divider_distance,
                                 float min_height_threshold, float clipping_height, float min_point_distance,
                                 float reclass_distance_threshold)
{
  sensor_height_ = sensor_height;                              // meters, 2.68
  general_max_slope_ = general_max_slope;                      // degrees, 3.0
  local_max_slope_ = local_max_slope;                          // degrees, 5.0
  radial_divider_angle_ = radial_divider_angle;                // distance in rads between dividers, 0.1
  concentric_divider_distance_ = concentric_divider_distance;  // distance in meters between concentric divisions,0.01
  min_height_threshold_ =
      min_height_threshold;            // minimum height threshold regardless the slope, useful for close points, 0.35
  clipping_height_ = clipping_height;  // the points higher than this will be removed from the input cloud., 0.3
  min_point_distance_ = min_point_distance;  // minimum distance from the origin to consider a point as valid, 1.85
  reclass_distance_threshold_ =
      reclass_distance_threshold;  // distance between points at which re classification will occur, 0.175

  radial_dividers_num_ = 0;
  concentric_dividers_num_ = 0;

  // GlobalVariable::UI_PARA[5]
}

template <typename PointT>
PointIndices RayGroundFilter::compute(typename PointCloud<PointT>::Ptr input)
{
  // Model   |   Horizontal   |   Vertical   | FOV(Vertical)    degrees / rads
  //----------------------------------------------------------
  // HDL-64  |0.08-0.35(0.32) |     0.4      |  -24.9 <=x<=2.0   (26.9  / 0.47)
  // HDL-32  |     0.1-0.4    |     1.33     |  -30.67<=x<=10.67 (41.33 / 0.72)
  // VLP-16  |     0.1-0.4    |     2.0      |  -15.0<=x<=15.0   (30    / 0.52)
  // VLP-16HD|     0.1-0.4    |     1.33     |  -10.0<=x<=10.0   (20    / 0.35)

  radial_dividers_num_ = ceil(360 / radial_divider_angle_);

  typename pcl::PointCloud<PointT>::Ptr clipped_cloud_ptr(new pcl::PointCloud<PointT>);

  // remove points above certain point
  ClipCloud(input, clipping_height_, clipped_cloud_ptr);

  // remove closer points than a threshold
  typename pcl::PointCloud<PointT>::Ptr filtered_cloud_ptr(new pcl::PointCloud<PointT>);
  RemovePointsUpTo(clipped_cloud_ptr, min_point_distance_, filtered_cloud_ptr);

  PointCloudXYZIRTColor organized_points;
  std::vector<pcl::PointIndices> radial_division_indices;
  std::vector<pcl::PointIndices> closest_indices;
  std::vector<PointCloudXYZIRTColor> radial_ordered_clouds;

  radial_dividers_num_ = ceil(360 / radial_divider_angle_);

  ConvertXYZIToRTZColor(filtered_cloud_ptr, organized_points, radial_division_indices, radial_ordered_clouds);

  pcl::PointIndices ground_indices, no_ground_indices;

  ClassifyPointCloud(radial_ordered_clouds, ground_indices, no_ground_indices);

  return ground_indices;
}

template PointIndices RayGroundFilter::compute<PointXYZ>(typename PointCloud<PointXYZ>::Ptr input);

/*
 * TODO PointXYZ for RayGround
 *
template
PointIndices
RayGroundFilter::compute<PointXYZI> (typename PointCloud<PointXYZI>::Ptr input);
*/

/*!
 *
 * @param[in] in_cloud Input Point Cloud to be organized in radial segments
 * @param[out] out_organized_points Custom Point Cloud filled with XYZRTZColor data
 * @param[out] out_radial_divided_indices Indices of the points in the original cloud for each radial segment
 * @param[out] out_radial_ordered_clouds Vector of Points Clouds, each element will contain the points ordered
 */
void RayGroundFilter::ConvertXYZIToRTZColor(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud,
                                            PointCloudXYZIRTColor& out_organized_points,
                                            std::vector<pcl::PointIndices>& out_radial_divided_indices,
                                            std::vector<PointCloudXYZIRTColor>& out_radial_ordered_clouds)
{
  out_organized_points.resize(in_cloud->points.size());
  out_radial_divided_indices.clear();
  out_radial_divided_indices.resize(radial_dividers_num_);
  out_radial_ordered_clouds.resize(radial_dividers_num_);

  for (size_t i = 0; i < in_cloud->points.size(); i++)
  {
    PointXYZIRTColor new_point;
    double radius =
        (float)sqrt(in_cloud->points[i].x * in_cloud->points[i].x + in_cloud->points[i].y * in_cloud->points[i].y);
    double theta = (float)atan2(in_cloud->points[i].y, in_cloud->points[i].x) * 180 / M_PI;
    if (theta < 0)
    {
      theta += 360;
    }

    size_t radial_div = (size_t)floor(theta / radial_divider_angle_);
    size_t concentric_div = (size_t)floor(fabs(radius / concentric_divider_distance_));

    new_point.point = in_cloud->points[i];
    new_point.radius = radius;
    new_point.theta = theta;
    new_point.radial_div = radial_div;
    new_point.concentric_div = concentric_div;
    new_point.original_index = i;

    out_organized_points[i] = new_point;

    // radial divisions
    out_radial_divided_indices[radial_div].indices.push_back(i);

    out_radial_ordered_clouds[radial_div].push_back(new_point);
  }

  // order radial points on each division
#pragma omp for
  for (size_t i = 0; i < radial_dividers_num_; i++)
  {
    std::sort(out_radial_ordered_clouds[i].begin(), out_radial_ordered_clouds[i].end(),
              [](const PointXYZIRTColor& a, const PointXYZIRTColor& b) { return a.radius < b.radius; });
  }
}

/*!
 * Classifies Points in the PointCoud as Ground and Not Ground
 * @param in_radial_ordered_clouds Vector of an Ordered PointsCloud ordered by radial distance from the origin
 * @param out_ground_indices Returns the indices of the points classified as ground in the original PointCloud
 * @param out_no_ground_indices Returns the indices of the points classified as not ground in the original PointCloud
 */
void RayGroundFilter::ClassifyPointCloud(std::vector<PointCloudXYZIRTColor>& in_radial_ordered_clouds,
                                         pcl::PointIndices& out_ground_indices,
                                         pcl::PointIndices& out_no_ground_indices)
{
  out_ground_indices.indices.clear();
  out_no_ground_indices.indices.clear();
#pragma omp for
  for (size_t i = 0; i < in_radial_ordered_clouds.size(); i++)  // sweep through each radial division
  {
    float prev_radius = 0.f;
    float prev_height = -sensor_height_;
    bool prev_ground = false;
    bool current_ground = false;
    for (size_t j = 0; j < in_radial_ordered_clouds[i].size(); j++)  // loop through each point in the radial div
    {
      float points_distance = in_radial_ordered_clouds[i][j].radius - prev_radius;
      float height_threshold = tan(DEG2RAD(local_max_slope_)) * points_distance;
      float current_height = in_radial_ordered_clouds[i][j].point.z;
      float general_height_threshold = tan(DEG2RAD(general_max_slope_)) * in_radial_ordered_clouds[i][j].radius;

      // for points which are very close causing the height threshold to be tiny, set a minimum value
      if (points_distance > concentric_divider_distance_ && height_threshold < min_height_threshold_)
      {
        height_threshold = min_height_threshold_;
      }

      // check current point height against the LOCAL threshold (previous point)
      if (current_height <= (prev_height + height_threshold) && current_height >= (prev_height - height_threshold))
      {
        // Check again using general geometry (radius from origin) if previous points wasn't ground
        if (!prev_ground)
        {
          if (current_height <= (-sensor_height_ + general_height_threshold) &&
              current_height >= (-sensor_height_ - general_height_threshold))
          {
            current_ground = true;
          }
          else
          {
            current_ground = false;
          }
        }
        else
        {
          current_ground = true;
        }
      }
      else
      {
        // check if previous point is too far from previous one, if so classify again
        if (points_distance > reclass_distance_threshold_ && (current_height <= (-sensor_height_ + height_threshold) &&
                                                              current_height >= (-sensor_height_ - height_threshold)))
        {
          current_ground = true;
        }
        else
        {
          current_ground = false;
        }
      }

      if (current_ground)
      {
        out_ground_indices.indices.push_back(in_radial_ordered_clouds[i][j].original_index);
        prev_ground = true;
      }
      else
      {
        out_no_ground_indices.indices.push_back(in_radial_ordered_clouds[i][j].original_index);
        prev_ground = false;
      }

      prev_radius = in_radial_ordered_clouds[i][j].radius;
      prev_height = in_radial_ordered_clouds[i][j].point.z;
    }
  }
}

/*!
 * Removes the points higher than a threshold
 * @param in_cloud_ptr PointCloud to perform Clipping
 * @param in_clip_height Maximum allowed height in the cloud
 * @param out_clipped_cloud_ptr Resultung PointCloud with the points removed
 */
void RayGroundFilter::ClipCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, double in_clip_height,
                                pcl::PointCloud<pcl::PointXYZ>::Ptr out_clipped_cloud_ptr)
{
  pcl::ExtractIndices<pcl::PointXYZ> extractor;
  extractor.setInputCloud(in_cloud_ptr);
  pcl::PointIndices indices;

#pragma omp for
  for (size_t i = 0; i < in_cloud_ptr->points.size(); i++)
  {
    if (in_cloud_ptr->points[i].z > in_clip_height)
    {
      indices.indices.push_back(i);
    }
  }
  extractor.setIndices(boost::make_shared<pcl::PointIndices>(indices));
  extractor.setNegative(true);  // true removes the indices, false leaves only the indices
  extractor.filter(*out_clipped_cloud_ptr);
}

/*!
 * Removes points up to a certain distance in the XY Plane
 * @param in_cloud_ptr Input PointCloud
 * @param in_min_distance Minimum valid distance, points closer than this will be removed.
 * @param out_filtered_cloud_ptr Resulting PointCloud with the invalid points removed.
 */
void RayGroundFilter::RemovePointsUpTo(const pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, double in_min_distance,
                                       pcl::PointCloud<pcl::PointXYZ>::Ptr out_filtered_cloud_ptr)
{
  pcl::ExtractIndices<pcl::PointXYZ> extractor;
  extractor.setInputCloud(in_cloud_ptr);
  pcl::PointIndices indices;

#pragma omp for
  for (size_t i = 0; i < in_cloud_ptr->points.size(); i++)
  {
    if (sqrt(in_cloud_ptr->points[i].x * in_cloud_ptr->points[i].x +
             in_cloud_ptr->points[i].y * in_cloud_ptr->points[i].y) < in_min_distance)
    {
      indices.indices.push_back(i);
    }
  }
  extractor.setIndices(boost::make_shared<pcl::PointIndices>(indices));
  extractor.setNegative(true);  // true removes the indices, false leaves only the indices
  extractor.filter(*out_filtered_cloud_ptr);
}
