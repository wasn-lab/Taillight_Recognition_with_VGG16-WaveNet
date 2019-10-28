#ifndef __IMU_UTILS_H__
#define __IMU_UTILS_H__

#include <iostream>
#include "camera_params.h"
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

namespace imu_distance
{
static constexpr double fov_w = 120.;
static constexpr double fov_h = fov_w * (camera::raw_image_height - 1) / (camera::raw_image_width - 1);
static constexpr double degs_per_px = fov_w / (camera::raw_image_width - 1);

double degree_to_radian(const double x);

void x2d_to_yaw(double& yaw_rad, const double x2d_pivot);

void y2d_to_pitch(double& pitch_rad, const double y2d_pivot);

// v: a vector in 3D space
// k: a unit vector describing the axis of rotation
// theta: the angle (in radians) that v rotates around k
glm::dvec3 rotate(const glm::dvec3& v, const glm::dvec3& k, double theta);

double angle_between(glm::vec3 a, glm::vec3 b, glm::vec3 origin);

};  // namespace

#endif  // __IMU_UTILS_H__