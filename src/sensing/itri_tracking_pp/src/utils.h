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
void rotate3(float a[][3], const unsigned int n, const float radians);

void translate(float a[][2], const unsigned int n, const float x_pivot, const float y_pivot);
void translate3(float a[][3], const unsigned int n, const float x_pivot, const float y_pivot, const float z_pivot);

void set_PoseRPY32(PoseRPY32& out, const PoseRPY32 in);
void set_Point32(Point32& out, const Point32 in);

void swap_Point32(Point32& A, Point32& B);
void convert_Point32_to_Point(geometry_msgs::Point& out, const Point32 in);
Point32 add_two_Point32s(const Point32 A, const Point32 B);

float compute_scalar_projection_A_onto_B(const float Ax, const float Ay, const float Az,  //
                                         const float Bx, const float By, const float Bz);

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

void set_ColorRGBA(std_msgs::ColorRGBA& out, const std_msgs::ColorRGBA in);
void set_ColorRGBA(std_msgs::ColorRGBA& c, const float r, const float g, const float b, const float a);

void set_config(const MarkerConfig& in, MarkerConfig& out);
}  // namespace tpp

#endif  // __UTILS_H__