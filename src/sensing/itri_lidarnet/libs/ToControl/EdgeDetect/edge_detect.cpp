#include "edge_detect.h"

pcl::PointCloud<PointXYZI> getContour(const pcl::PointCloud<PointXYZIL>::Ptr input_cloud, const float theta_sample,
                                      const float range_low_bound, const float range_up_bound)
{
  float delta_theta = 360 / theta_sample;

  pcl::PointCloud<PointXYZI> contour_cloud;

  contour_cloud.width = theta_sample;
  contour_cloud.height = 1;
  contour_cloud.points.resize(contour_cloud.width * contour_cloud.height);
  contour_cloud.is_dense = false;

  for (size_t i = 0; i < contour_cloud.points.size(); i++)
  {
    contour_cloud.points[i].x = NAN;
    contour_cloud.points[i].y = NAN;
    contour_cloud.points[i].z = 0;
    contour_cloud.points[i].intensity = 0;
  }

  for (size_t i = 0; i < input_cloud->points.size(); i++)
  {
    // float r_point = hypot(input_cloud->points[i].x, input_cloud->points[i].y);
    float r_point = input_cloud->points[i].x * input_cloud->points[i].x +
                    input_cloud->points[i].y * input_cloud->points[i].y;  // taking summation of squares (omit square
                                                                          // root) as the range for comparison

    if (r_point > range_low_bound * range_low_bound && r_point < range_up_bound * range_up_bound)
    {
      float theta_point = atan2(input_cloud->points[i].y, input_cloud->points[i].x) * R2D;  // deg

      if (theta_point < 0)
      {
        theta_point += 360;
      }

      int indx_theta = round(theta_point / delta_theta);
      // float r_contour = hypot(contour_cloud.points[indx_theta].x, contour_cloud.points[indx_theta].y);
      float r_contour = contour_cloud.points[indx_theta].x * contour_cloud.points[indx_theta].x +
                        contour_cloud.points[indx_theta].y * contour_cloud.points[indx_theta].y;

      if (std::isnan(r_contour) || r_point < r_contour)
      {
        contour_cloud.points[indx_theta].x = input_cloud->points[i].x;
        contour_cloud.points[indx_theta].y = input_cloud->points[i].y;
      }
    }
  }

  return contour_cloud;
}
