#include "alignment_utils.h"
#include "alignment_params.h"
#include "math.h"

namespace alignment
{
cv::Point map_pcd_ground_point_to_image_point(const pcl::PointXYZ& pcd_point, const int cam_sn)
{
  cv::Point result(-1, -1);
  int height_in_centermeter = pcd_point.z * 100;
  if ((height_in_centermeter < -330) || (height_in_centermeter > -300))
  {
    return result;
  }
  else
  {
    return map_pcd_point_to_image_point(pcd_point, cam_sn);
  }
}

cv::Point map_pcd_point_to_image_point(const pcl::PointXYZ& pcd_point, const int cam_sn)
{
  cv::Point result(-1, -1);

  const auto& invR_T = alignment::get_invR_T(cam_sn);
  const auto& invT_T = alignment::get_invT_T(cam_sn);
  const double theta_max = alignment::get_theta_max(cam_sn);
  const double theta_min = alignment::get_theta_min(cam_sn);
  const double phi_max = alignment::get_phi_max(cam_sn);
  const double phi_min = alignment::get_phi_min(cam_sn);
  const auto& alig_CameraMat_ = alignment::get_alignment_camera_mat(cam_sn);
  const auto& alig_DistCoeff_ = alignment::get_alignment_dist_coeff_mat(cam_sn);

  double dist_raw = sqrt(pcd_point.x * pcd_point.x + pcd_point.y * pcd_point.y + pcd_point.z * pcd_point.z);

  double theta = acos(pcd_point.z / dist_raw) / 2 / M_PI * 360;
  double phi = atan2(pcd_point.y, pcd_point.x) / 2 / M_PI * 360;

  if (phi < phi_max && phi > phi_min && theta < theta_max && theta > theta_min)
  {
    // This calibration tool assumes that the Velodyne is installed with the default order of axes for the Velodyne
    // sensor.
    // X axis points to the front
    // Y axis points to the left
    // Z axis points upwards

    cv::Point2d imagepoint(-1.0, -1.0);
    cv::Mat point(1, 3, CV_64F);
    point.at<double>(0) = (double)pcd_point.x;
    point.at<double>(1) = (double)pcd_point.y;
    point.at<double>(2) = (double)pcd_point.z;

    point = point * invR_T + invT_T;

    double dist = (double)(point.at<double>(0) * point.at<double>(0)) +
                  (double)(point.at<double>(1) * point.at<double>(1)) +
                  (double)(point.at<double>(2) * point.at<double>(2));
    dist = sqrt(dist);

    double tmpx = point.at<double>(0) / point.at<double>(2);
    double tmpy = point.at<double>(1) / point.at<double>(2);
    double r2 = tmpx * tmpx + tmpy * tmpy;
    double tmpdist = 1 + alig_DistCoeff_.at<double>(0) * r2 + alig_DistCoeff_.at<double>(1) * r2 * r2 +
                     alig_DistCoeff_.at<double>(4) * r2 * r2 * r2;

    imagepoint.x = tmpx * tmpdist + 2 * alig_DistCoeff_.at<double>(2) * tmpx * tmpy +
                   alig_DistCoeff_.at<double>(3) * (r2 + 2 * tmpx * tmpx);
    imagepoint.y = tmpy * tmpdist + alig_DistCoeff_.at<double>(2) * (r2 + 2 * tmpy * tmpy) +
                   2 * alig_DistCoeff_.at<double>(3) * tmpx * tmpy;

    imagepoint.x = alig_CameraMat_.at<double>(0, 0) * imagepoint.x + alig_CameraMat_.at<double>(0, 2);
    imagepoint.y = alig_CameraMat_.at<double>(1, 1) * imagepoint.y + alig_CameraMat_.at<double>(1, 2);
    int px = int(imagepoint.x + 0.5);
    int py = int(imagepoint.y + 0.5);
    if ((px < 0) || (px >= alignment::camera_image_width) || (py < 0) || (py >= alignment::camera_image_height))
    {
      px = -1;
      py = -1;
    }
    result.x = px;
    result.y = py;
  }

  return result;
}

int map_pcd_ground_to_image_points(pcl::PointCloud<pcl::PointXYZ> release_cloud, const int cam_sn)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud = release_cloud.makeShared();
  int num_valid_points = 0;

  for (size_t i = 0; i < cloud->size(); i++)
  {
    const auto imagepoint = map_pcd_ground_point_to_image_point(cloud->points[i], cam_sn);
    if ((imagepoint.x >= 0) && (imagepoint.y >= 0))
    {
      num_valid_points++;
    }
  }
  return num_valid_points;
}

int map_pcd_to_image_points(pcl::PointCloud<pcl::PointXYZ> release_cloud, const int cam_sn)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud = release_cloud.makeShared();
  int num_valid_points = 0;

  // cv::Mat M_LIDAR_2_MID = cv::Mat::zeros(1208, 1920, CV_32FC3); //for spatial alignment
  for (size_t i = 0; i < cloud->size(); i++)
  {
    const auto imagepoint = map_pcd_point_to_image_point(cloud->points[i], cam_sn);
    if ((imagepoint.x >= 0) && (imagepoint.y >= 0))
    {
      num_valid_points++;
    }
  }
  return num_valid_points;
}

double* get_dist_by_image_point(pcl::PointCloud<pcl::PointXYZ> release_cloud, int image_x, int image_y, int cam_sn)
{
  static double r[3];

  bool search_value = false;

  const auto& invR_T = alignment::get_invR_T(cam_sn);
  const auto& invT_T = alignment::get_invT_T(cam_sn);
  const double theta_max = alignment::get_theta_max(cam_sn);
  const double theta_min = alignment::get_theta_min(cam_sn);
  const double phi_max = alignment::get_phi_max(cam_sn);
  const double phi_min = alignment::get_phi_min(cam_sn);
  const auto& alig_CameraMat_ = alignment::get_alignment_camera_mat(cam_sn);
  const auto& alig_DistCoeff_ = alignment::get_alignment_dist_coeff_mat(cam_sn);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud = release_cloud.makeShared();

  // cv::Mat M_LIDAR_2_MID = cv::Mat::zeros(1208, 1920, CV_32FC3); //for spatial alignment
  for (size_t i = 0; i < cloud->size(); i++)
  {
    double dist_raw = sqrt(cloud->points[i].x * cloud->points[i].x + cloud->points[i].y * cloud->points[i].y +
                           cloud->points[i].z * cloud->points[i].z);

    double theta = acos(cloud->points[i].z / dist_raw) / 2 / M_PI * 360;
    double phi = atan2(cloud->points[i].y, cloud->points[i].x) / 2 / M_PI * 360;

    if (phi < phi_max && phi > phi_min && theta < theta_max && theta > theta_min)
    {
      // This calibration tool assumes that the Velodyne is installed with the default order of axes for the Velodyne
      // sensor.
      // X axis points to the front
      // Y axis points to the left
      // Z axis points upwards

      cv::Mat point(1, 3, CV_64F);
      point.at<double>(0) = (double)cloud->points[i].x;
      point.at<double>(1) = (double)cloud->points[i].y;
      point.at<double>(2) = (double)cloud->points[i].z;

      point = point * invR_T + invT_T;

      double dist = (double)(point.at<double>(0) * point.at<double>(0)) +
                    (double)(point.at<double>(1) * point.at<double>(1)) +
                    (double)(point.at<double>(2) * point.at<double>(2));
      dist = sqrt(dist);

      double tmpx = point.at<double>(0) / point.at<double>(2);
      double tmpy = point.at<double>(1) / point.at<double>(2);
      double r2 = tmpx * tmpx + tmpy * tmpy;
      double tmpdist = 1 + alig_DistCoeff_.at<double>(0) * r2 + alig_DistCoeff_.at<double>(1) * r2 * r2 +
                       alig_DistCoeff_.at<double>(4) * r2 * r2 * r2;
      cv::Point2d imagepoint;

      imagepoint.x = tmpx * tmpdist + 2 * alig_DistCoeff_.at<double>(2) * tmpx * tmpy +
                     alig_DistCoeff_.at<double>(3) * (r2 + 2 * tmpx * tmpx);
      imagepoint.y = tmpy * tmpdist + alig_DistCoeff_.at<double>(2) * (r2 + 2 * tmpy * tmpy) +
                     2 * alig_DistCoeff_.at<double>(3) * tmpx * tmpy;

      imagepoint.x = alig_CameraMat_.at<double>(0, 0) * imagepoint.x + alig_CameraMat_.at<double>(0, 2);
      imagepoint.y = alig_CameraMat_.at<double>(1, 1) * imagepoint.y + alig_CameraMat_.at<double>(1, 2);

      if (imagepoint.x > 0 && imagepoint.y > 0)
      {
        int px = int(imagepoint.x + 0.5);
        int py = int(imagepoint.y + 0.5);

        double range_x_min, range_x_max, range_y_min, range_y_max;
        if (cam_sn == 7 || cam_sn == 9)
        {
          range_x_min = (image_x > 80) ? (image_x - 80) : 0;
          range_x_max = ((image_x + 80) < 1920) ? (image_x + 80) : 1920;

          range_y_min = (image_y > 10) ? (image_y - 10) : 0;
          range_y_max = ((image_y + 10) < 1208) ? (image_y + 10) : 1208;
        }
        else
        {
          range_x_min = (image_x > 10) ? (image_x - 10) : 0;
          range_x_max = ((image_x + 10) < 1920) ? (image_x + 10) : 1920;

          range_y_min = (image_y > 65) ? (image_y - 65) : 0;
          range_y_max = ((image_y + 65) < 1208) ? (image_y + 65) : 1208;
        }

        if (px == image_x && py == image_y)
        {
          r[0] = (double)cloud->points[i].x;
          r[1] = (double)cloud->points[i].y;
          r[2] = (double)cloud->points[i].z;

          search_value = true;
          return r;
        }
        else
        {
          if (px > range_x_min && px < range_x_max && search_value == false)
          {
            if (py > range_y_min && py < range_y_max)
            {
              double now = sqrt((px - image_x) * (px - image_x) + (py - image_y) * (py - image_y));
              double pre =
                  sqrt((alignment::camera_image_width - image_x) * (alignment::camera_image_width - image_x) +
                       (alignment::camera_image_height - image_y) * (alignment::camera_image_height - image_y));
              if (now < pre)
              {
                r[0] = (double)cloud->points[i].x;
                r[1] = (double)cloud->points[i].y;
                r[2] = (double)cloud->points[i].z;
              }
            }
          }
        }
      }
    }
  }
  return r;
}

int get_image_x_min(const int image_x, const int cam_sn)
{
  if (cam_sn == 7 || cam_sn == 9)
  {
    return (image_x > 80) ? (image_x - 80) : 0;
  }
  else
  {
    return (image_x > 10) ? (image_x - 10) : 0;
  }
}

int get_image_x_max(const int image_x, const int cam_sn)
{
  if (cam_sn == 7 || cam_sn == 9)
  {
    return ((image_x + 80) < alignment::camera_image_width) ? (image_x + 80) : alignment::camera_image_width;
  }
  else
  {
    return ((image_x + 10) < alignment::camera_image_width) ? (image_x + 10) : alignment::camera_image_width;
  }
}

int get_image_y_min(const int image_y, const int cam_sn)
{
  if (cam_sn == 7 || cam_sn == 9)
  {
    return (image_y > 10) ? (image_y - 10) : 0;
  }
  else
  {
    return (image_y > 65) ? (image_y - 65) : 0;
  }
}

int get_image_y_max(const int image_y, const int cam_sn)
{
  if (cam_sn == 7 || cam_sn == 9)
  {
    return ((image_y + 10) < alignment::camera_image_height) ? (image_y + 10) : alignment::camera_image_height;
  }
  else
  {
    return ((image_y + 65) < alignment::camera_image_height) ? (image_y + 65) : alignment::camera_image_height;
  }
}
};  // namespace
