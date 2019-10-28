#include "imu_utils.h"

namespace imu_distance
{
double degree_to_radian(const double x)
{
  return x * 0.01745329252;
}

void x2d_to_yaw(double& yaw_rad, const double x2d_pivot)
{
  yaw_rad = degree_to_radian(0.5 * fov_w - x2d_pivot * degs_per_px);
}

void y2d_to_pitch(double& pitch_rad, const double y2d_pivot)
{
  pitch_rad = degree_to_radian(y2d_pivot * degs_per_px - 0.5 * fov_h);
}

// v: a vector in 3D space
// k: a unit vector describing the axis of rotation
// theta: the angle (in radians) that v rotates around k
glm::dvec3 rotate(const glm::dvec3& v, const glm::dvec3& k, double theta)
{
  // std::cout << "Rotating " << glm::to_string(v) << " " << theta << " radians around " << glm::to_string(k) << "..."
  //           << std::endl;

  double cos_theta = cos(theta);
  double sin_theta = sin(theta);

  glm::dvec3 rotated = (v * cos_theta) + (glm::cross(k, v) * sin_theta) + (k * glm::dot(k, v)) * (1 - cos_theta);

  // std::cout << "Rotated: " << glm::to_string(rotated) << std::endl;

  return rotated;
}

double angle_between(glm::vec3 a, glm::vec3 b, glm::vec3 origin)
{
  glm::vec3 da = glm::normalize(a - origin);
  glm::vec3 db = glm::normalize(b - origin);
  return glm::acos(glm::dot(da, db));
}

};  // namespace