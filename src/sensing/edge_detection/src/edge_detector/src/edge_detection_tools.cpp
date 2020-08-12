#include "headers.h"
#include "edge_detection_tools.h"

void saveFrame(const string filename_prefix, const int frame_num, const CloudConstPtr& frame_cloud)
{
  string filename = filename_prefix;
  stringstream ss;
  ss << frame_num;
  filename = filename + "_" + ss.str() + ".pcd";
  std::cout << filename;
  pcl::io::savePCDFileASCII(filename, *frame_cloud);
}

void getEdgeContour(const boost::shared_ptr<pcl::PointCloud<PointT> >& cloud,
                    boost::shared_ptr<pcl::PointCloud<PointT> >& right_edge_ptscloud,
                    boost::shared_ptr<pcl::PointCloud<PointT> >& left_edge_ptscloud, int theta_sample,
                    const float length, float threshold_width)
{
  ///@ theta_sample sample number between +-length @ length +- y length of LiDAR @  threshold_width width of x
  // double y_;
  double delta_length = length / theta_sample;

  boost::shared_ptr<pcl::PointCloud<PointT> > right_edge_(new pcl::PointCloud<PointT>);
  boost::shared_ptr<pcl::PointCloud<PointT> > left_edge_(new pcl::PointCloud<PointT>);

  right_edge_->width = theta_sample * 2;
  right_edge_->height = 1;
  right_edge_->points.resize(right_edge_->width * right_edge_->height);

  left_edge_->width = theta_sample * 2;
  left_edge_->height = 1;
  left_edge_->points.resize(left_edge_->width * left_edge_->height);

  for (int j = -theta_sample; j < theta_sample; ++j)
  {
    float y_ = j * delta_length;  // deg

    right_edge_->points[j + theta_sample].x = threshold_width;
    right_edge_->points[j + theta_sample].y = y_;
    right_edge_->points[j + theta_sample].z = 0;
    right_edge_->points[j + theta_sample].intensity = 150;
    right_edge_->is_dense = false;

    left_edge_->points[j + theta_sample].x = -threshold_width;
    left_edge_->points[j + theta_sample].y = y_;
    left_edge_->points[j + theta_sample].z = 0;
    left_edge_->points[j + theta_sample].intensity = 150;
    left_edge_->is_dense = false;
  }

  //#pragma omp parallel for
  for (size_t i = 0; i < cloud->points.size(); ++i)
  {
    if (fabs(cloud->points[i].y) < length)
    {
      /// calculate y index

      int length_indx = round(cloud->points[i].y / delta_length);

      /// if cloud_x > 0
      if (cloud->points[i].x > 0)
      {
        /// if cloud_x < right_data[y_index]
        if (fabs(cloud->points[i].x) < fabs(right_edge_->points[length_indx + theta_sample].x))
        {
          right_edge_->points[length_indx + theta_sample].x = cloud->points[i].x;
        }
      }
      /// if cloud_x < 0
      else
      {
        /// if cloud_x > left_data[y_index]
        if (cloud->points[i].x > left_edge_->points[length_indx + theta_sample].x)
        {
          left_edge_->points[length_indx + theta_sample].x = cloud->points[i].x;
        }
      }
    }
  }
  *right_edge_ptscloud = *right_edge_;
  *left_edge_ptscloud = *left_edge_;
}

void getContour(const boost::shared_ptr<pcl::PointCloud<PointT> > cloud,
                boost::shared_ptr<pcl::PointCloud<PointT> > cloud_contour, int theta_sample,
                const float threshold_radius)
{
  double theta;
  double delta_theta = 360 / theta_sample;
  cloud_contour->width = theta_sample;
  cloud_contour->height = 1;
  cloud_contour->points.resize(cloud_contour->width * cloud_contour->height);

  for (size_t j = 0; j < cloud_contour->points.size(); ++j)
  {
    theta = j * delta_theta;     // deg
    float r = threshold_radius;  // m

    cloud_contour->points[j].x = r * cos(theta / DEG_PER_RAD);
    cloud_contour->points[j].y = r * sin(theta / DEG_PER_RAD);
    cloud_contour->points[j].z = 0;
    cloud_contour->points[j].intensity = 150;
  }

  for (size_t i = 0; i < cloud->points.size(); ++i)
  {
    double r_point = hypot(cloud->points[i].x, cloud->points[i].y);
    double theta_2 = atan2(cloud->points[i].y, cloud->points[i].x) * DEG_PER_RAD;  // deg
    if (theta_2 < 0)
    {
      theta_2 += 360;
    }
    int indx_theta = round(theta_2 / delta_theta);
    double r_2 = hypot(cloud_contour->points[indx_theta].x, cloud_contour->points[indx_theta].y);
    if (r_point < r_2)
    {
      cloud_contour->points[indx_theta].x = cloud->points[i].x;
      cloud_contour->points[indx_theta].y = cloud->points[i].y;
    }
  }
}

void getContourV2(const boost::shared_ptr<pcl::PointCloud<PointT> >& cloud,
                  const boost::shared_ptr<pcl::PointCloud<PointT> >& cloud_ground,
                  boost::shared_ptr<pcl::PointCloud<PointT> >& cloud_contour,
                  boost::shared_ptr<pcl::PointCloud<PointT> >& cloud_innercontour, std::vector<float>& contour_distance,
                  const double& theta_sample, const float& threshold_radius, bool calculate_interior)
{
  // const float nap = std::numeric_limits<float>::quiet_NaN();
  double theta;
  double delta_theta = 360 / theta_sample;
  cloud_contour->width = theta_sample;
  cloud_contour->height = 1;
  cloud_contour->points.resize(cloud_contour->width * cloud_contour->height);
  cloud_contour->is_dense = false;
  *cloud_innercontour = *cloud_ground;
  cloud_innercontour->is_dense = false;
  float r = threshold_radius;  // m

  for (size_t j = 0; j < cloud_contour->points.size(); ++j)
  {
    theta = j * delta_theta;  // deg

    cloud_contour->points[j].x = r * cos(theta / DEG_PER_RAD);
    cloud_contour->points[j].y = r * sin(theta / DEG_PER_RAD);
    cloud_contour->points[j].z = 0;
    cloud_contour->points[j].intensity = 1;
  }
  for (size_t i = 0; i < cloud->points.size(); ++i)
  {
    // double r_point = hypot( cloud->points[i].x, cloud->points[i].y); //平方根
    double r_point =
        (cloud->points[i].x) * (cloud->points[i].x) + (cloud->points[i].y) * (cloud->points[i].y);  //平方根

    double theta_2 = atan2(cloud->points[i].y, cloud->points[i].x) * DEG_PER_RAD;  // deg

    if (theta_2 < 0)
    {
      theta_2 += 360;
    }
    int indx_theta = round(theta_2 / delta_theta);
    // double r_contour = hypot( cloud_contour->points[indx_theta].x, cloud_contour->points[indx_theta].y);
    double r_contour = (cloud_contour->points[indx_theta].x) * (cloud_contour->points[indx_theta].x) +
                       (cloud_contour->points[indx_theta].y) * (cloud_contour->points[indx_theta].y);

    if (r_point < r_contour)
    {
      cloud_contour->points[indx_theta].x = cloud->points[i].x;
      cloud_contour->points[indx_theta].y = cloud->points[i].y;
    }
  }

  if (calculate_interior)
  {
    for (size_t i = 0; i < cloud_innercontour->points.size(); ++i)
    {
      // double r_point = hypot( cloud_ground->points[i].x, cloud_ground->points[i].y); //平方根
      double r_point = (cloud_ground->points[i].x) * (cloud_ground->points[i].x) +
                       (cloud_ground->points[i].y) * (cloud_ground->points[i].y);  //平方根

      double theta_2 = atan2(cloud_ground->points[i].y, cloud_ground->points[i].x) * DEG_PER_RAD;
      if (theta_2 < 0)
      {
        theta_2 += 360;
      }
      //int indx_theta = round(theta_2 / delta_theta);
      // double r_contour = hypot( cloud_contour->points[indx_theta].x, cloud_contour->points[indx_theta].y);
      double r_contour = (cloud_contour->points[i].x) * (cloud_contour->points[i].x) +
                         (cloud_contour->points[i].y) * (cloud_contour->points[i].y);  //平方根

      if (r_point < r_contour - 1)
      {
        std::vector<int> indices;
        cloud_innercontour->points[i].x = cloud_ground->points[i].x;
        cloud_innercontour->points[i].y = cloud_ground->points[i].y;
        cloud_innercontour->points[i].z = cloud_ground->points[i].z;
      }
      else
      {
        cloud_innercontour->points[i].x = cloud_innercontour->points[i].y = cloud_innercontour->points[i].z =
            std::numeric_limits<float>::quiet_NaN();
        ;
      }
    }
  }

  // Nan outside contour ************
  for (size_t j = 0; j < cloud_contour->points.size(); ++j)
  {
    std::vector<float> tmp_contour;
    double r_contour = hypot(cloud_contour->points[j].x, cloud_contour->points[j].y);

    // double r_contour =
    // (cloud_contour->points[j].x)*(cloud_contour->points[j].x)+(cloud_contour->points[j].y)*(cloud_contour->points[j].y);

    if (r < r_contour + 0.1 || r_contour < 0.7)
    {
      cloud_contour->points[j].x = cloud_contour->points[j].y = cloud_contour->points[j].z =
          std::numeric_limits<float>::quiet_NaN();
      contour_distance.push_back(r);
    }
    else
    {
      contour_distance.push_back(r_contour);
    }
  }
}

bool approximateProgressiveMorphological(const pcl::PointCloud<PointT>::ConstPtr input, pcl::PointIndicesPtr& ground)
{
  pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<100> >::Ptr fpfh(
      new pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<100> >);

  if (input->size() < 2)
  {
    std::cerr << "Could not estimate a ProgressiveMorphological model for the given dataset." << std::endl;
    return false;
  }
  pcl::ApproximateProgressiveMorphologicalFilter<PointT> apmf;
  apmf.setInputCloud(input);

  // BoChun's Parameters
  // apmf.setCellSize(1);
  // apmf.setBase(3);  // 0.1
  // apmf.setExponential(false);
  // apmf.setMaxWindowSize(1.0);    // 3.0
  // apmf.setSlope(0.2);            // 0.2 //1.0f
  // apmf.setInitialDistance(0.3);  // 0.12 //0.1
  // apmf.setMaxDistance(0.6);      // 0.2 //0.3

  // Wayne's Parameters
  apmf.setCellSize(0.35);
  apmf.setBase(2);  // 0.1
  apmf.setExponential(false);
  apmf.setMaxWindowSize(1.0);    // 3.0
  apmf.setSlope(0.9);            // 0.2 //1.0f
  apmf.setInitialDistance(0.32);  // 0.12 //0.1
  apmf.setMaxDistance(0.34);      // 0.2 //0.3
  
  apmf.setNumberOfThreads(2);
  apmf.extract(ground->indices);
  return true;
}

bool extractParallelLine(const boost::shared_ptr<pcl::PointCloud<PointT> >& input_cloud,
                         boost::shared_ptr<pcl::PointCloud<PointT> >& output_line,
                         boost::shared_ptr<pcl::PointCloud<PointT> >& output_non_line, float distance_threshhold,
                         Eigen::VectorXf& line_coefficients)

{
  //** RANSAC parallel_ine extract ***************************
  if (input_cloud->points.size() > 1)  // bigger than 1 RANSAC
  {
    pcl::PointIndices::Ptr tmp_cloud_line_inliers(new pcl::PointIndices);
    pcl::SampleConsensusModelParallelLine<PointT>::Ptr model_line(
        new pcl::SampleConsensusModelParallelLine<PointT>(input_cloud));
    Eigen::Vector3f ax(0, 1, 0);
    model_line->setAxis(ax);
    model_line->setEpsAngle(pcl::deg2rad(10.0));
    pcl::RandomSampleConsensus<PointT> ransac_line(model_line);
    ransac_line.setDistanceThreshold(distance_threshhold);
    ransac_line.computeModel();
    ransac_line.getModelCoefficients(line_coefficients);
    ransac_line.getInliers(tmp_cloud_line_inliers->indices);

    // Extract the outliers
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(input_cloud);
    extract.setIndices(tmp_cloud_line_inliers);
    extract.setNegative(true);
    extract.filter(*output_non_line);

    extract.setInputCloud(input_cloud);
    extract.setIndices(tmp_cloud_line_inliers);
    extract.setNegative(false);
    extract.filter(*output_line);
    return true;
  }
  else
  {
    return false;
  }
}

void radiusFilter(boost::shared_ptr<pcl::PointCloud<PointT> >& input_cloud,
                  boost::shared_ptr<pcl::PointCloud<PointT> >& cloud_filtered, double radius, int min_pts)
{
  pcl::RadiusOutlierRemoval<PointT> radiusF;
  radiusF.setInputCloud(input_cloud);
  radiusF.setRadiusSearch(radius);  // unit:m
  radiusF.setMinNeighborsInRadius(min_pts);
  radiusF.filter(*cloud_filtered);  // apply filter
}


void coFilter(boost::shared_ptr<pcl::PointCloud<PointT> >& input_cloud,
              boost::shared_ptr<pcl::PointCloud<PointT> >& cloud_filtered, const std::string& co, float min_value,
              float max_value, bool negative)
{
  pcl::PassThrough<PointT> pass;
  pass.setInputCloud(input_cloud);
  pass.setFilterFieldName(co), pass.setFilterLimits(min_value, max_value);
  pass.setFilterLimitsNegative(negative);
  pass.filter(*cloud_filtered);
}

void iFilter(boost::shared_ptr<pcl::PointCloud<PointT> >& input_cloud,
             boost::shared_ptr<pcl::PointCloud<PointT> >& cloud_filtered, float i_min, float i_max, bool i_negtive)

{
  pcl::PassThrough<PointT> pass_i;
  pass_i.setInputCloud(input_cloud);
  pass_i.setFilterFieldName("intensity");
  pass_i.setFilterLimits(i_min, i_max);
  pass_i.setFilterLimitsNegative(i_negtive);
  pass_i.filter(*cloud_filtered);
}

void iFilterAug(boost::shared_ptr<pcl::PointCloud<PointT> >& input_cloud,
                boost::shared_ptr<pcl::PointCloud<PointT> >& cloud_filtered, float i_min, float i_max, bool i_negtive,
                float value)

{
  pcl::PassThrough<PointT> pass_i;
  pass_i.setInputCloud(input_cloud);
  pass_i.setFilterFieldName("intensity");
  pass_i.setFilterLimits(i_min, i_max);
  pass_i.setFilterLimitsNegative(i_negtive);
  pass_i.filter(*cloud_filtered);

  for (size_t i = 0; i < cloud_filtered->points.size(); ++i)
  {
    cloud_filtered->points[i].intensity = value;
  }
}
pcl::PointCloud<pcl::PointXYZI> hollow_removal(boost::shared_ptr<pcl::PointCloud<PointT> >& input, double Xmin,
                                               double Xmax, double Ymin, double Ymax, double Zmin, double Zmax,
                                               double Xmin_outlier, double Xmax_outlier, double Ymin_outlier,
                                               double Ymax_outlier, double Zmin_outlier, double Zmax_outlier)
{
  PointCloud<PointT> buff = *input;

  int k = 0;
  for (size_t i = 0; i < buff.size(); i++)
  {
    if ((buff.points[i].x < Xmax_outlier && buff.points[i].x > Xmin_outlier) &&
        (buff.points[i].y < Ymax_outlier && buff.points[i].y > Ymin_outlier) &&
        (buff.points[i].z < Zmax_outlier && buff.points[i].z > Zmin_outlier))
    {
      if (!((buff.points[i].x < Xmax && buff.points[i].x > Xmin) &&
            (buff.points[i].y < Ymax && buff.points[i].y > Ymin) &&
            (buff.points[i].z < Zmax && buff.points[i].z > Zmin)))
      {
        buff.points[k] = buff.points[i];
        k++;
      }
    }
  }
  buff.resize(k);

  return buff;
}

void boxFilter(boost::shared_ptr<pcl::PointCloud<PointT> >& cloudIn,
               boost::shared_ptr<pcl::PointCloud<PointT> >& cloudOut, float min_x, float min_y, float min_z,
               float max_x, float max_y, float max_z, bool setNegative)

{
  Eigen::Vector4f minPoint;
  minPoint[0] = min_x;  // define minimum point x
  minPoint[1] = min_y;  // define minimum point y
  minPoint[2] = min_z;  // define minimum point z
  Eigen::Vector4f maxPoint;
  maxPoint[0] = max_x;  // define max point x
  maxPoint[1] = max_y;  // define max point y
  maxPoint[2] = max_z;  // define max point z

  pcl::CropBox<PointT> cropFilter;
  cropFilter.setInputCloud(cloudIn);
  cropFilter.setNegative(setNegative);
  cropFilter.setMin(minPoint);
  cropFilter.setMax(maxPoint);
  cropFilter.filter(*cloudOut);
}

void projectToZPlane(const boost::shared_ptr<pcl::PointCloud<PointT> >& input_cloud,
                     boost::shared_ptr<pcl::PointCloud<PointT> >& output_2D_cloud)
{
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
  // ax+by+cz+d=0, where a=b=d=0, and c=1
  coefficients->values.resize(4);
  coefficients->values[0] = 0;
  coefficients->values[1] = 0;
  coefficients->values[2] = 1.0;
  coefficients->values[3] = 0;

  pcl::ProjectInliers<PointT> proj;
  proj.setModelType(pcl::SACMODEL_PLANE);
  proj.setInputCloud(input_cloud);
  proj.setModelCoefficients(coefficients);
  proj.filter(*output_2D_cloud);
}
void extractIndice(const boost::shared_ptr<pcl::PointCloud<PointT> >& input_cloud,
                   boost::shared_ptr<pcl::PointCloud<PointT> >& output_cloud, pcl::PointIndicesPtr& ground_Indice,
                   bool negative)
{
  pcl::ExtractIndices<PointT> extract;
  extract.setInputCloud(input_cloud);
  extract.setIndices(ground_Indice);
  extract.setNegative(negative);
  extract.filter(*output_cloud);
}

void voxelDownsample(boost::shared_ptr<pcl::PointCloud<PointT> >& input_cloud,
                     boost::shared_ptr<pcl::PointCloud<PointT> >& output_cloud, float leafsize)
{
  pcl::VoxelGrid<PointT> vg;
  vg.setInputCloud(input_cloud);
  vg.setLeafSize(leafsize, leafsize, leafsize);
  vg.filter(*output_cloud);
}

float originToLine(const Eigen::VectorXf& model_coefficients)
{
  Eigen::Vector3f origin(0, 0, 0);
  Eigen::Vector3f line_pt(model_coefficients[0], model_coefficients[1], model_coefficients[2]);
  Eigen::Vector3f line_dir(model_coefficients[3], model_coefficients[4], model_coefficients[5]);
  line_dir.normalize();                                           // norm of line dir
  return sqrt((line_pt - origin).cross(line_dir).squaredNorm());  // d = |a x B|/|B|
}

void setInputCloud(const CloudConstPtr input, boost::shared_ptr<pcl::PointCloud<PointT> > release_cloud)
{
  *release_cloud = *input;
}
float confidenceDgree(boost::shared_ptr<pcl::PointCloud<PointT> >& input_num,
                      boost::shared_ptr<pcl::PointCloud<PointT> >& input_den)
{
  float num = input_num->points.size();
  float den = input_den->points.size();

  return num / den;
}
