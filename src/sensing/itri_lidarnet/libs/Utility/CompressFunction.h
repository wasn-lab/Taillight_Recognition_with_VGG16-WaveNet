#ifndef COMPRESSFUNCTION_H_
#define COMPRESSFUNCTION_H_

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/extract_indices.h>

using namespace pcl;

class CompressFunction
{
  public:
    CompressFunction ();
    ~CompressFunction ();

    pcl::visualization::Camera
    CamPara (double A,
             double B,
             double C,
             double D,
             double E,
             double F,
             double G,
             double H,
             double I,
             double J,
             double K,
             double L,
             double M,
             double N,
             double O,
             double P,
             double Q);


};


#endif
