// 2018 Industrial Technology Research Insitute

#include <time.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iterator>
#include <numeric>
#include <algorithm>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

typedef struct _convtools convtools_t;
struct _convtools
{
  Mat plot;
  Mat arbit;
  Mat real;
};

typedef struct _transmat transmat_t;
struct _transmat
{
  double R[9];
  double iR[9];
  double t[3];
  double K[9];
  double iK[9];
};

typedef struct _imagedim imagedim_t;
struct _imagedim
{
  int width;
  int height;
};
typedef struct _d3point d3point_t;
struct _d3point
{
  double x;
  double y;
  double z;
};
typedef struct _d2point d2point_t;
struct _d2point
{
  int u;
  int v;
  int d;
};

void vector_subtract_3d(const double v1[3], const double v2[3], double result[3]);
double matrix_determinant_3x3d(const double m[9]);
void matrix_inverse_3x3d(const double m[9], double inverse[9]);
void matrix_vector_multiply_3x3_3d(const double m[9], const double v[3], double result[3]);
void vector_add_3d(const double v1[3], const double v2[3], double result[3]);
// void load_config(char CONFIGfile_name[256], double t_mat[3], double R_mat[9], double K_mat[3]);
// You can remove the following two functions for real-time implementation
// void visualize(Mat& image, int num_pc, int** PIXEL);
// void read_lidar(const char *lidar_name, int index_lidar, const char *lidar_type, double** &GNSS, int** &PIXEL, int*
// &GNSS_refc, int& num_points, transmat_t trans, imagedim_t imgdim);
//=======================================================================
void point3Dto2D(d3point_t input, d2point_t& output, transmat_t trans, imagedim_t imgdim, convtools_t& ctool);
void point2Dto3D(d2point_t input, d3point_t& output, transmat_t trans, imagedim_t imgdim, convtools_t ctool);
