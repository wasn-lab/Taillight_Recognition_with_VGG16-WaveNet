/*
   CREATER: ICL U300
   DATE: July, 2019
 */

#ifndef __ALIGNMENT_PARAMS_H__
#define __ALIGNMENT_PARAMS_H__
#include "opencv2/core/mat.hpp"
#include <string>

namespace alignment
{
constexpr int camera_image_width = 1920;
constexpr int camera_image_height = 1208;
const cv::Mat& get_invR_T(const int cam_sn);
const cv::Mat& get_invT_T(const int cam_sn);
const cv::Mat& get_alignment_camera_mat(const int cam_sn);
const cv::Mat& get_alignment_dist_coeff_mat(const int cam_sn);
double get_phi_min(const int cam_sn);
double get_phi_max(const int cam_sn);
double get_theta_min(const int cam_sn);
double get_theta_max(const int cam_sn);
std::string get_lidar_topic_by_cam_sn(const int cam_sn);
};  // namespace

#endif  // __ALIGNMENT_PARAMS_H__
