#include "utils.h"

namespace tpp
{
float divide(const float dividend, const float divisor)
{
  if (divisor == 0)
  {
    throw std::runtime_error("Math error: Attempted to divide by Zero\n");
  }

  return (dividend / divisor);
}

void increase_uint(unsigned int& x)
{
  if (x == 4294967295)
  {
    x = 1;
  }
  else
  {
    x++;
  }
}

double clock_to_milliseconds(clock_t num_ticks)
{
  // units/(units/time) => time (seconds) * 1000 = milliseconds
  return (num_ticks / (double)CLOCKS_PER_SEC) * 1000.0;
}

float squared_euclidean_distance(const float x1, const float y1, const float x2, const float y2)
{
  return std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2);
}

float euclidean_distance(const float a, const float b)
{
  return std::sqrt(std::pow(a, 2) + std::pow(b, 2));
}

float euclidean_distance3(const float a, const float b, const float c)
{
  return std::sqrt(std::pow(a, 2) + std::pow(b, 2) + std::pow(c, 2));
}

// km/h to m/s
float kmph_to_mps(const float kmph)
{
  return (0.277778f * kmph);
}

// m/s to km/h
float mps_to_kmph(const float mps)
{
  return (3.6f * mps);
}

void rotate(float a[][2], const unsigned int n, const float radians)
{
  unsigned int i = 0;
  while (i < n)
  {
    float x_shifted = a[i][0];
    float y_shifted = a[i][1];

    float cos_ = std::cos(radians);
    float sin_ = std::sin(radians);

    a[i][0] = x_shifted * cos_ - y_shifted * sin_;
    a[i][1] = x_shifted * sin_ + y_shifted * cos_;

    i++;
  }
}

void rotate_eigen3(float& out_x, float& out_y, const float in_x, const float in_y, const float in_ang_rad)
{
  Eigen::Vector2f v;
  v << in_x, in_y;

  Eigen::Rotation2Df t(in_ang_rad);
  t.toRotationMatrix();

  Eigen::Vector2f rotated_v = t * v;

  out_x = rotated_v(0);
  out_y = rotated_v(1);
}

void rotate3(float a[][3], const unsigned int n, const float radians)
{
  unsigned int i = 0;
  while (i < n)
  {
    float x_shifted = a[i][0];
    float y_shifted = a[i][1];

    float cos_ = std::cos(radians);
    float sin_ = std::sin(radians);

    a[i][0] = x_shifted * cos_ - y_shifted * sin_;
    a[i][1] = x_shifted * sin_ + y_shifted * cos_;

    i++;
  }
}

void translate(float a[][2], const unsigned int n, const float x_pivot, const float y_pivot)
{
  unsigned int i = 0;
  while (i < n)
  {
    a[i][0] += x_pivot;
    a[i][1] += y_pivot;
    i++;
  }
}

void translate3(float a[][3], const unsigned int n, const float x_pivot, const float y_pivot, const float z_pivot)
{
  unsigned int i = 0;
  while (i < n)
  {
    a[i][0] += x_pivot;
    a[i][1] += y_pivot;
    a[i][2] += z_pivot;
    i++;
  }
}

void set_PoseRPY32(PoseRPY32& out, const PoseRPY32 in)
{
  out.x = in.x;
  out.y = in.y;
  out.z = in.z;

  out.roll = in.roll;
  out.pitch = in.pitch;
  out.yaw = in.yaw;
}

void set_MyPoint32(MyPoint32& out, const MyPoint32 in)
{
  out.x = in.x;
  out.y = in.y;
  out.z = in.z;
}

void swap_MyPoint32(MyPoint32& A, MyPoint32& B)
{
  MyPoint32 tmp;

  tmp.x = A.x;
  tmp.y = A.y;
  tmp.z = A.z;

  A.x = B.x;
  A.y = B.y;
  A.z = B.z;

  B.x = tmp.x;
  B.y = tmp.y;
  B.z = tmp.z;
}

void convert_MyPoint32_to_Point(geometry_msgs::Point& out, const MyPoint32 in)
{
  out.x = (double)in.x;
  out.y = (double)in.y;
  out.z = (double)in.z;
}

MyPoint32 add_two_MyPoint32s(const MyPoint32 A, const MyPoint32 B)
{
  MyPoint32 C;
  C.x = A.x + B.x;
  C.y = A.y + B.y;
  C.z = A.z + B.z;
  return C;
}

float compute_scalar_projection_A_onto_B(const float Ax, const float Ay, const float Az,  //
                                         const float Bx, const float By, const float Bz)
{
  if (Bx == 0 && By == 0 && Bz == 0)
  {
    return 0;
  }

  float dot = Ax * Bx + Ay * By + Az * Bz;
  float unit_vec = euclidean_distance3(Bx, By, Bz);
  return (float)(dot / unit_vec);
}

void transform_point_abs2rel(const float x_abs, const float y_abs, const float z_abs,  //
                             float& x_rel, float& y_rel, float& z_rel,                 //
                             const float ego_x_abs, const float ego_y_abs, const float ego_z_abs,
                             const float ego_heading)
{
  float point[1][3] = { { x_abs, y_abs, z_abs } };
  translate3(point, 1, -ego_x_abs, -ego_y_abs, -ego_z_abs);

#if EIGEN3_ROTATION
  rotate_eigen3(x_rel, y_rel, point[0][0], point[0][1], -ego_heading);
#else
  rotate3(point, 1, -ego_heading);

  x_rel = point[0][0];
  y_rel = point[0][1];
#endif

  z_rel = point[0][2];
}

void transform_point_rel2abs(const float x_rel, const float y_rel, const float z_rel,  //
                             float& x_abs, float& y_abs, float& z_abs,                 //
                             const float ego_x_abs, const float ego_y_abs, const float ego_z_abs,
                             const float ego_heading)
{
  float point[1][3] = { { x_rel, y_rel, z_rel } };
#if EIGEN3_ROTATION
  rotate_eigen3(point[0][0], point[0][1], point[0][0], point[0][1], ego_heading);
#else
  rotate3(point, 1, ego_heading);
#endif
  translate3(point, 1, ego_x_abs, ego_y_abs, ego_z_abs);

  x_abs = point[0][0];
  y_abs = point[0][1];
  z_abs = point[0][2];
}

void transform_vector_abs2rel(const float vx_abs, const float vy_abs, float& vx_rel, float& vy_rel,
                              const float ego_heading)
{
#if EIGEN3_ROTATION
  rotate_eigen3(vx_rel, vy_rel, vx_abs, vy_abs, -ego_heading);
#else
  float vec[1][2] = { { vx_abs, vy_abs } };

  rotate(vec, 1, -ego_heading);

  vx_rel = vec[0][0];
  vy_rel = vec[0][1];
#endif
}

void transform_vector_rel2abs(const float vx_rel, const float vy_rel, float& vx_abs, float& vy_abs,
                              const float ego_heading)
{
#if EIGEN3_ROTATION
  rotate_eigen3(vx_abs, vy_abs, vx_rel, vy_rel, ego_heading);
#else
  float vec[1][2] = { { vx_rel, vy_rel } };

  rotate(vec, 1, ego_heading);

  vx_abs = vec[0][0];
  vy_abs = vec[0][1];
#endif
}

void set_ColorRGBA(std_msgs::ColorRGBA& out, const std_msgs::ColorRGBA in)
{
  out.r = in.r;
  out.g = in.g;
  out.b = in.b;
  out.a = in.a;
}

void set_ColorRGBA(std_msgs::ColorRGBA& c, const float r, const float g, const float b, const float a)
{
  c.r = r;
  c.g = g;
  c.b = b;
  c.a = a;
}

void set_config(const MarkerConfig& in, MarkerConfig& out)
{
  out.pub_pp = in.pub_pp;
  out.pub_vel = in.pub_vel;
  out.pub_id = in.pub_id;
  out.pub_speed = in.pub_speed;

  out.lifetime_sec = in.lifetime_sec;
  out.module_pubtime_sec = in.module_pubtime_sec;

  out.show_classid = in.show_classid;
  out.show_tracktime = in.show_tracktime;
  out.show_source = in.show_source;
  out.show_distance = in.show_distance;
  out.show_absspeed = in.show_absspeed;
  out.show_pp = in.show_pp;

  set_ColorRGBA(out.color, in.color);
}

void quaternion_to_rpy(double& roll, double& pitch, double& yaw, const double q_x, const double q_y, const double q_z,
                       const double q_w)
{
  tf::Quaternion q(q_x, q_y, q_z, q_w);
  tf::Matrix3x3 m(q);

  m.getRPY(roll, pitch, yaw);
}
}  // namespace tpp
