#ifndef __UTILS_H__
#define __UTILS_H__

#include <ctime>
#include <cmath>
#include <msgs/PointXYZ.h>
#include <std_msgs/ColorRGBA.h>
#include "tpp_base.h"

namespace tpp
{
float divide(const float dividend, const float divisor);

template <typename T>
void assign_value_cannot_zero(T& out, const T in)
{
  T epsilon = (T)0.00001;
  out = (in == (T)0) ? epsilon : in;
}

void increase_uint(unsigned int& x);

double clock_to_milliseconds(const clock_t num_ticks);

float squared_euclidean_distance(const float x1, const float y1, const float x2, const float y2);

float euclidean_distance(const float a, const float b);

float euclidean_distance3(const float a, const float b, const float c);

// km/h to m/s
float kmph_to_mps(const float kmph);

// m/s to km/h
float mps_to_kmph(const float mps);

void rotate(float a[][2], const unsigned int n, const float radians);
void rotate_eigen3(float& out_x, float& out_y, const float in_x, const float in_y, const float in_ang_rad);
void rotate3(float a[][3], const unsigned int n, const float radians);  // actually rotate x y only

void translate(float a[][2], const unsigned int n, const float x_pivot, const float y_pivot);
void translate3(float a[][3], const unsigned int n, const float x_pivot, const float y_pivot, const float z_pivot);

void set_PoseRPY32(PoseRPY32& out, const PoseRPY32 in);
void set_MyPoint32(MyPoint32& out, const MyPoint32 in);

void swap_MyPoint32(MyPoint32& A, MyPoint32& B);
void convert_MyPoint32_to_Point(geometry_msgs::Point& out, const MyPoint32 in);
MyPoint32 add_two_MyPoint32s(const MyPoint32 A, const MyPoint32 B);

void transform_point_abs2rel(const float x_abs, const float y_abs, const float z_abs,  //
                             float& x_rel, float& y_rel, float& z_rel,                 //
                             const float ego_x_abs, const float ego_y_abs, const float ego_z_abs,
                             const float ego_heading);

void transform_point_rel2abs(const float x_rel, const float y_rel, const float z_rel,  //
                             float& x_abs, float& y_abs, float& z_abs,                 //
                             const float ego_x_abs, const float ego_y_abs, const float ego_z_abs,
                             const float ego_heading);

void transform_vector_abs2rel(const float vx_abs, const float vy_abs, float& vx_rel, float& vy_rel,
                              const float ego_heading);

void transform_vector_rel2abs(const float vx_rel, const float vy_rel, float& vx_abs, float& vy_abs,
                              const float ego_heading);

void quaternion_to_rpy(double& roll, double& pitch, double& yaw, const double q_x, const double q_y, const double q_z,
                       const double q_w);
}  // namespace tpp

#endif  // __UTILS_H__