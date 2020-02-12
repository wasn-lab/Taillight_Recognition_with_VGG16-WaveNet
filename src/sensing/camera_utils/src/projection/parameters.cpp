#include "parameters.h"
CalibrateParameters::CalibrateParameters(double t_x, double t_y, double t_z, double d_x, double d_y, double d_z,
                                         unsigned int f, unsigned int u, unsigned int v, string t, string o_t)
  : t_x_double_(t_x)
  , t_y_double_(t_y)
  , t_z_double_(t_z)
  , degree_x_double_(d_x)
  , degree_y_double_(d_y)
  , degree_z_double_(d_z)
  , focal_length_uint_(f)
  , center_point_u_uint_(u)
  , center_point_v_uint_(v)
  , camera_topic_string_(t)
  , camera_object_topic_string_(o_t)
{
}

CalibrateParameters::CalibrateParameters()
{
}

CalibrateParameters::~CalibrateParameters()
{
}

void CalibrateParameters::set_t_matrix(double x, double y, double z)
{
  t_x_double_ = x;
  t_y_double_ = y;
  t_z_double_ = z;
}

void CalibrateParameters::set_r_matrix(double x, double y, double z)
{
  degree_x_double_ = x;
  degree_y_double_ = y;
  degree_z_double_ = z;
}

void CalibrateParameters::set_focal_length(unsigned int f)
{
  focal_length_uint_ = f;
}

void CalibrateParameters::set_center_point(unsigned int u, unsigned int v)
{
  center_point_u_uint_ = u;
  center_point_v_uint_ = v;
}

double CalibrateParameters::get_t_x()
{
  return t_x_double_;
}

double CalibrateParameters::get_t_y()
{
  return t_y_double_;
}

double CalibrateParameters::get_t_z()
{
  return t_z_double_;
}

double CalibrateParameters::get_degree_x()
{
  return degree_x_double_;
}

double CalibrateParameters::get_degree_y()
{
  return degree_y_double_;
}

double CalibrateParameters::get_degree_z()
{
  return degree_z_double_;
}

unsigned int CalibrateParameters::get_focal_length()
{
  return focal_length_uint_;
}

unsigned int CalibrateParameters::get_center_point_u()
{
  return center_point_u_uint_;
}

unsigned int CalibrateParameters::get_center_point_v()
{
  return center_point_v_uint_;
}

string CalibrateParameters::get_camera_topic()
{
  return camera_topic_string_;
}

string CalibrateParameters::get_camera_obj_topic()
{
  return camera_object_topic_string_;
}
