#include "LShape.h"

LShape::LShape ()
{
  random_points_ = 80;
  slope_dist_thres_ = 2.0;
  num_points_thres_ = 10;
  sensor_height_ = 2.64 - 1.4;

  // Assuming pointcloud x and y range within roi_m_: 0 < x, y < roi_m_
  // Short for region of interest in meters
  roi_m_ = 120;//120

  // Scale roi_m*roi_m_ to pic_scale_ times: will end up cv::Mat<roi_m_*pic_scale_, roi_m_*pic_scale_>
  // in order to make fitting algorithm work
  pic_scale_ = 15;//15

}

LShape::~LShape ()
{
}

void
LShape::setInputCloud (const pcl::PointCloud<pcl::PointXYZ> input)
{
  cloud_3d = input;
}

void
LShape::Compute (std::vector<pcl::PointXYZ> &out_cube,
                 pcl::PointXYZ &out_centroid,
                 pcl::PointXYZ &out_minPoint,
                 pcl::PointXYZ &out_maxPoint)
{

  // calculating offset so that projecting pointcloud into cv::mat
  cv::Mat m (pic_scale_ * roi_m_, pic_scale_ * roi_m_, CV_8UC1, cv::Scalar (0));
  cv::Point2f tmp_pointcloud_point (cloud_3d[0].x, cloud_3d[0].y);
  cv::Point2f tmp_pointcloud_offset (roi_m_ / 2, roi_m_ / 2);
  cv::Point2f tmp_offset_pointcloud_point = tmp_pointcloud_point + tmp_pointcloud_offset;
  cv::Point tmp_pic_point = tmp_offset_pointcloud_point * pic_scale_;

  int tmp_init_pic_x = tmp_pic_point.x;
  int tmp_init_pic_y = pic_scale_ * roi_m_ - tmp_pic_point.y;

  cv::Point tmp_init_pic_point (tmp_init_pic_x, tmp_init_pic_y);
  cv::Point tmp_init_offset_vec (roi_m_ * pic_scale_ / 2, roi_m_ * pic_scale_ / 2);
  cv::Point offset_init_pic_point = tmp_init_offset_vec - tmp_init_pic_point;

  int num_points = cloud_3d.size ();
  std::vector<cv::Point> point_vec (num_points);
  std::vector<cv::Point2f> pointcloud_frame_points (4);

  // init variables
  cv::Point2f min_m_p (0, 0);
  cv::Point2f max_m_p (0, 0);
  float min_m = std::numeric_limits<float>::max ();
  float max_m = std::numeric_limits<float>::lowest ();

  for (int i_point = 0; i_point < num_points; i_point++)
  {
    const float p_x = cloud_3d[i_point].x;
    const float p_y = cloud_3d[i_point].y;

    // cast (roi_m_/2 < x,y < roi_m_/2) into (0 < x,y < roi_m_)
    cv::Point2f pointcloud_point (p_x, p_y);
    cv::Point2f pointcloud_offset_vec (roi_m_ / 2, roi_m_ / 2);
    cv::Point2f offset_pointcloud_point = pointcloud_point + pointcloud_offset_vec;
    // cast (roi_m_)m*(roi_m_)m into  pic_scale_
    cv::Point scaled_point = offset_pointcloud_point * pic_scale_;
    // cast into image coordinate
    int pic_x = scaled_point.x;
    int pic_y = pic_scale_ * roi_m_ - scaled_point.y;
    // offset so that the object would be locate at the center
    cv::Point pic_point (pic_x, pic_y);
    cv::Point offset_point = pic_point + offset_init_pic_point;

    // Make sure points are inside the image size
    if (offset_point.x > (pic_scale_ * roi_m_) || offset_point.x<0 || offset_point.y < 0 || offset_point.y> (pic_scale_ * roi_m_))
    {
      continue;
    }
    // cast the pointcloud into cv::mat
    m.at<uchar> (offset_point.y, offset_point.x) = 255;
    point_vec[i_point] = offset_point;
    // calculate min and max slope for x1, x3(edge points)
    float delta_m = p_y / p_x;
    if (delta_m < min_m)
    {
      min_m = delta_m;
      min_m_p.x = p_x;
      min_m_p.y = p_y;
    }

    if (delta_m > max_m)
    {
      max_m = delta_m;
      max_m_p.x = p_x;
      max_m_p.y = p_y;
    }
  }
  if (max_m == std::numeric_limits<float>::lowest () || min_m == std::numeric_limits<float>::max ())
  {

  }
  else
  {
    // L shape fitting parameters
    cv::Point2f dist_vec = max_m_p - min_m_p;
    float slope_dist = sqrt (dist_vec.x * dist_vec.x + dist_vec.y * dist_vec.y);
    float slope = (max_m_p.y - min_m_p.y) / (max_m_p.x - min_m_p.x);

    // random variable
    std::mt19937_64 mt;
    mt.seed (ros::Time::now ().toSec ());
    // mt.seed(0);
    std::uniform_int_distribution<> rand_points (0, num_points - 1);

    // start l shape fitting for car like object
    if (slope_dist > slope_dist_thres_ && num_points > num_points_thres_)
    {
      float max_dist = 0;
      cv::Point2f max_p (0, 0);

      // get max distance from random sampling points
      for (int i = 0; i < random_points_; i++)
      {
        int p_ind = rand_points (mt);
        assert(p_ind >= 0 && p_ind < (cloud_3d.size () - 1));
        cv::Point2f p_i (cloud_3d[p_ind].x, cloud_3d[p_ind].y);

        // from equation of distance between line and point
        float dist = std::abs (slope * p_i.x - 1 * p_i.y + max_m_p.y - slope * max_m_p.x) / std::sqrt (slope * slope + 1);
        if (dist > max_dist)
        {
          max_dist = dist;
          max_p = p_i;
        }
      }
      // vector adding
      cv::Point2f max_m_vec = max_m_p - max_p;
      cv::Point2f min_m_vec = min_m_p - max_p;
      cv::Point2f last_p = max_p + max_m_vec + min_m_vec;

      pointcloud_frame_points[0] = min_m_p;
      pointcloud_frame_points[1] = max_p;
      pointcloud_frame_points[2] = max_m_p;
      pointcloud_frame_points[3] = last_p;
    }
    else
    {
      // MinAreaRect fitting
      cv::RotatedRect rect_info = cv::minAreaRect (point_vec);
      cv::Point2f rect_points[4];
      rect_info.points (rect_points);
      // covert points back to lidar coordinate
      getPointsInPointcloudFrame (rect_points, pointcloud_frame_points, offset_init_pic_point);
    }

    double PosePX, PosePY, PosePZ;

    // update output_object pose
    updateCpFromPoints (pointcloud_frame_points, PosePX, PosePY, PosePZ);

    // update pointcloud_frame_points to make it right angle bbox
    toRightAngleBBox (pointcloud_frame_points);

    double PoseOX, PoseOY, PoseOZ, PoseOW;
    double DimX, DimY, DimZ;  //L,W,H
    double Yaw;
    // update output_object dimensions
    updateDimentionAndEstimatedAngle (pointcloud_frame_points, PoseOX, PoseOY, PoseOZ, PoseOW, DimX, DimY, DimZ,Yaw);

    //ITRI----------------------------------------------

    Eigen::Vector4f Centroid3D;
    //pcl::compute3DCentroid (cloud_3d, Centroid3D);
    //out_centroid.x = Centroid3D[0];
    //out_centroid.y = Centroid3D[1];
    //out_centroid.z = Centroid3D[2];

    pcl::getMinMax3D (cloud_3d, out_minPoint, out_maxPoint);

    pcl::PointCloud<pcl::PointXYZ> point8;
    point8.resize (8);
    point8.points.at (0) = (pcl::PointXYZ (out_minPoint.x, out_minPoint.y, out_minPoint.z));
    point8.points.at (1) = (pcl::PointXYZ (out_minPoint.x, out_minPoint.y, out_maxPoint.z));
    point8.points.at (2) = (pcl::PointXYZ (out_maxPoint.x, out_minPoint.y, out_maxPoint.z));
    point8.points.at (3) = (pcl::PointXYZ (out_maxPoint.x, out_minPoint.y, out_minPoint.z));
    point8.points.at (4) = (pcl::PointXYZ (out_minPoint.x, out_maxPoint.y, out_minPoint.z));
    point8.points.at (5) = (pcl::PointXYZ (out_minPoint.x, out_maxPoint.y, out_maxPoint.z));
    point8.points.at (6) = (pcl::PointXYZ (out_maxPoint.x, out_maxPoint.y, out_maxPoint.z));
    point8.points.at (7) = (pcl::PointXYZ (out_maxPoint.x, out_maxPoint.y, out_minPoint.z));


    pcl::PointXYZ fff((point8.points.at (6).x + point8.points.at (0).x)/2,(point8.points.at (6).y + point8.points.at (0).y)/2,0);


    for(size_t i=0;i<8;i++){
      point8.points.at (i).x = point8.points.at (i).x-fff.x;
      point8.points.at (i).y = point8.points.at (i).y-fff.y;
      point8.points.at (i).z = point8.points.at (i).z-fff.z;
    }


    pcl::PointCloud<pcl::PointXYZ> buff;
/*
    Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity ();
    transform_2.translation () << PosePX-fff.x, PosePY-fff.y, PosePZ-fff.z;
    transform_2.rotate (Eigen::AngleAxisf (0, Eigen::Vector3f::UnitX ()));
    transform_2.rotate (Eigen::AngleAxisf (0, Eigen::Vector3f::UnitY ()));
    transform_2.rotate (Eigen::AngleAxisf (0, Eigen::Vector3f::UnitZ ()));
    pcl::transformPointCloud (point8, buff, transform_2);*/

    Eigen::Vector3f aaa(PosePX-fff.x, PosePY-fff.y, PosePZ-fff.z);
    Eigen::Quaternionf bbb(PoseOW,PoseOX, PoseOY, PoseOZ);

    pcl::transformPointCloud (point8,buff, aaa,bbb ,true);


    out_cube.resize (8);
    for(size_t i=0;i<8;i++){
      out_cube.at (i).x = buff.points[i].x+fff.x;
      out_cube.at (i).y = buff.points[i].y+fff.y;
      out_cube.at (i).z = buff.points[i].z+fff.z;
    }

  }

}

void
LShape::getPointsInPointcloudFrame (cv::Point2f rect_points[],
                                    std::vector<cv::Point2f>& pointcloud_frame_points,
                                    const cv::Point& offset_point)
{
  // loop 4 rect points
  for (int point_i = 0; point_i < 4; point_i++)
  {
    cv::Point2f offset_point_float (offset_point.x, offset_point.y);

    cv::Point2f reverse_offset_point = rect_points[point_i] - offset_point_float;
    // reverse from image coordinate to eucledian coordinate
    float r_x = reverse_offset_point.x;
    float r_y = pic_scale_ * roi_m_ - reverse_offset_point.y;
    cv::Point2f eucledian_coordinate_pic_point (r_x, r_y);
    // reverse to roi_m_*roi_m_ scale
    cv::Point2f offset_pointcloud_point = eucledian_coordinate_pic_point * float (1 / pic_scale_);
    // reverse from (0 < x,y < roi_m_) to (roi_m_/2 < x,y < roi_m_/2)
    cv::Point2f offset_vec_ (roi_m_ / 2, roi_m_ / 2);
    cv::Point2f pointcloud_point = offset_pointcloud_point - offset_vec_;
    pointcloud_frame_points[point_i] = pointcloud_point;
  }
}

void
LShape::updateCpFromPoints (const std::vector<cv::Point2f>& pointcloud_frame_points,
                            double&PoseX,
                            double&PoseY,
                            double&PoseZ)
{
  cv::Point2f p1 = pointcloud_frame_points[0];
  cv::Point2f p2 = pointcloud_frame_points[1];
  cv::Point2f p3 = pointcloud_frame_points[2];
  cv::Point2f p4 = pointcloud_frame_points[3];

  double s1 = ( (p4.x - p2.x) * (p1.y - p2.y) - (p4.y - p2.y) * (p1.x - p2.x)) / 2;
  double s2 = ( (p4.x - p2.x) * (p2.y - p3.y) - (p4.y - p2.y) * (p2.x - p3.x)) / 2;
  double cx = p1.x + (p3.x - p1.x) * s1 / (s1 + s2);
  double cy = p1.y + (p3.y - p1.y) * s1 / (s1 + s2);

  PoseX = cx;
  PoseY = cy;
  PoseZ = -sensor_height_ / 2;
}

void
LShape::toRightAngleBBox (std::vector<cv::Point2f>& pointcloud_frame_points)
{
  cv::Point2f p1 = pointcloud_frame_points[0];
  cv::Point2f p2 = pointcloud_frame_points[1];
  cv::Point2f p3 = pointcloud_frame_points[2];

  cv::Point2f vec1 (p2.x - p1.x, p2.y - p1.y);
  cv::Point2f vec2 (p3.x - p2.x, p3.y - p2.y);

  // from the equation of inner product
  double cos_theta = vec1.dot (vec2) / (norm (vec1) + norm (vec2));
  double theta = acos (cos_theta);
  double diff_theta = theta - M_PI / 2;

  if (std::abs (diff_theta) > 0.1)
  {
    double m1 = vec1.y / vec1.x;
    double b1 = p3.y - m1 * p3.x;
    double m2 = -1.0 / m1;
    double b2 = p2.y - (m2 * p2.x);

    double x = (b2 - b1) / (m1 - m2);
    double y = (b2 * m1 - b1 * m2) / (m1 - m2);

    double delta_x = x - p2.x;
    double delta_y = y - p2.y;

    pointcloud_frame_points[2].x = x;
    pointcloud_frame_points[2].y = y;
    pointcloud_frame_points[3].x = pointcloud_frame_points[0].x + delta_x;
    pointcloud_frame_points[3].y = pointcloud_frame_points[0].y + delta_y;
  }
}

void
LShape::updateDimentionAndEstimatedAngle (const std::vector<cv::Point2f>& pointcloud_frame_points,
                                          double&PoseOX,
                                          double&PoseOY,
                                          double&PoseOZ,
                                          double&PoseOW,
                                          double&DimX,
                                          double&DimY,
                                          double&DimZ,
                                          double&Yaw
                                          )
{
  // p1-p2 and p2-p3 is line segment, p1-p3 is diagonal
  cv::Point2f p1 = pointcloud_frame_points[0];
  cv::Point2f p2 = pointcloud_frame_points[1];
  cv::Point2f p3 = pointcloud_frame_points[2];

  cv::Point2f vec1 = p1 - p2;
  cv::Point2f vec2 = p3 - p2;
  double dist1 = norm (vec1);
  double dist2 = norm (vec2);
  double bb_yaw;
  // dist1 is length, dist2 is width
  if (dist1 > dist2)
  {
    bb_yaw = atan2 (p1.y - p2.y, p1.x - p2.x);
    DimX = dist1;
    DimY = dist2;
    DimZ = 2;
  }
  // dist1 is width, dist2 is length
  else
  {
    bb_yaw = atan2 (p3.y - p2.y, p3.x - p2.x);
    DimX = dist2;
    DimY = dist1;
    DimZ = 2;
  }

  Yaw = bb_yaw;
  // convert yaw to quartenion
  tf::Matrix3x3 obs_mat;
  obs_mat.setEulerYPR (bb_yaw, 0, 0);

  tf::Quaternion q_tf;
  obs_mat.getRotation (q_tf);
  PoseOX = q_tf.getX ();
  PoseOY = q_tf.getY ();
  PoseOZ = q_tf.getZ ();
  PoseOW = q_tf.getW ();
}
