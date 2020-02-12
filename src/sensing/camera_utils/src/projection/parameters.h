#ifndef PARAMETERS_H
#define PARAMETERS_H
#include <string>

using namespace std;

class CalibrateParameters
{
private:
  double t_x_double_;
  double t_y_double_;
  double t_z_double_;
  double degree_x_double_;
  double degree_y_double_;
  double degree_z_double_;
  unsigned int focal_length_uint_;
  unsigned int center_point_u_uint_;
  unsigned int center_point_v_uint_;
  string camera_topic_string_;
  string camera_object_topic_string_;

public:
  // constructor
  CalibrateParameters();
  CalibrateParameters(double t_x, double t_y, double t_z, double d_x, double d_y, double d_z, unsigned int f,
                      unsigned int u, unsigned int v, string t, string o_t);
  ~CalibrateParameters();

  // setter
  void set_t_matrix(double x, double y, double z);
  void set_r_matrix(double x, double y, double z);
  void set_focal_length(unsigned int f);
  void set_center_point(unsigned u, unsigned v);

  // getter
  double get_t_x();
  double get_t_y();
  double get_t_z();
  double get_degree_x();
  double get_degree_y();
  double get_degree_z();
  unsigned int get_focal_length();
  unsigned int get_center_point_u();
  unsigned int get_center_point_v();
  string get_camera_topic();
  string get_camera_obj_topic();
};
#endif
