#ifndef EDGE_DETECTION_TOOLS_H_
#define EDGE_DETECTION_TOOLS_H_

#include "headers.h"


void saveFrame (const string filename_prefix, const int frame_num, const CloudConstPtr& frame_cloud);

void
getEdgeContour (const boost::shared_ptr<pcl::PointCloud<PointT> > &cloud,
                boost::shared_ptr<pcl::PointCloud<PointT> > &right_edge_ptscloud,
                boost::shared_ptr<pcl::PointCloud<PointT> > &left_edge_ptscloud,
                int theta_sample, const float length, float threshold_width);

void
getContour (const boost::shared_ptr<pcl::PointCloud<PointT> > cloud,
            boost::shared_ptr<pcl::PointCloud<PointT> > cloud_contour,
            int theta_sample, const float threshold_radius);


void
getContourV2 (const boost::shared_ptr<pcl::PointCloud<PointT> > &cloud,
              const boost::shared_ptr<pcl::PointCloud<PointT> > &cloud_ground,
              boost::shared_ptr<pcl::PointCloud<PointT> > &cloud_contour,
              boost::shared_ptr<pcl::PointCloud<PointT> > &cloud_innercontour,
              std::vector<float> &contour_distance,
              const double &theta_sample, const float &threshold_radius,
              bool calculate_interior
              );


bool
approximateProgressiveMorphological (const pcl::PointCloud<PointT>::ConstPtr input,
                                     pcl::PointIndicesPtr &ground);


bool
extractParallelLine (  const boost::shared_ptr<pcl::PointCloud<PointT> > &input_cloud,
                       boost::shared_ptr<pcl::PointCloud<PointT> > &output_line,
                       boost::shared_ptr<pcl::PointCloud<PointT> > &output_non_line,
                       float distance_threshhold,
                       Eigen::VectorXf &line_coefficients);



void
coFilter (boost::shared_ptr<pcl::PointCloud<PointT> > &input_cloud,
          boost::shared_ptr<pcl::PointCloud<PointT> > &cloud_filtered,
          const std::string &co,
          float min_value, float max_value,
          bool negative);


void
iFilter (boost::shared_ptr<pcl::PointCloud<PointT> > &input_cloud,
         boost::shared_ptr<pcl::PointCloud<PointT> > &cloud_filtered,
         float i_min,float i_max,
         bool i_negtive);

pcl::PointCloud<pcl::PointXYZI>
hollow_removal (boost::shared_ptr<pcl::PointCloud<PointT> > & input,
                double Xmin,
                double Xmax,
                double Ymin,
                double Ymax,
                double Zmin,
                double Zmax,
                double Xmin_outlier,
                double Xmax_outlier,
                double Ymin_outlier,
                double Ymax_outlier,
                double Zmin_outlier,
                double Zmax_outlier);
void
boxFilter (boost::shared_ptr<pcl::PointCloud<PointT> > &cloudIn,
           boost::shared_ptr<pcl::PointCloud<PointT> > &cloudOut,
           float min_x,
           float min_y,
           float min_z,
           float max_x,
           float max_y,
           float max_z,
           bool setNegative

           );

void
iFilterAug (boost::shared_ptr<pcl::PointCloud<PointT> > &input_cloud,
            boost::shared_ptr<pcl::PointCloud<PointT> > &cloud_filtered,
            float i_min,float i_max,
            bool i_negtive, float value);

void
projectToZPlane (const boost::shared_ptr<pcl::PointCloud<PointT> > &input_cloud,
                 boost::shared_ptr<pcl::PointCloud<PointT> > &output_2D_cloud);


void
extractIndice (const boost::shared_ptr<pcl::PointCloud<PointT> > &input_cloud,
               boost::shared_ptr<pcl::PointCloud<PointT> > &output_cloud,
               pcl::PointIndicesPtr &ground_Indice,
               bool negative
               );


void
voxelDownsample (boost::shared_ptr<pcl::PointCloud<PointT> > &input_cloud,
                 boost::shared_ptr<pcl::PointCloud<PointT> > &output_cloud,
                 float leafsize);

float
originToLine (const Eigen::VectorXf &model_coefficients);

void
setInputCloud (const CloudConstPtr input, boost::shared_ptr<pcl::PointCloud<PointT> > release_cloud);

float
confidenceDgree (boost::shared_ptr<pcl::PointCloud<PointT> > &input_num,
                 boost::shared_ptr<pcl::PointCloud<PointT> > &input_den);




#endif /* EDGE_DETECTION_TOOLS_H_ */
