/*
   CREATER: ICL U300
   DATE: July, 2019
*/

#include "alignment_params.h"
#include <assert.h>

#define NUM_CAM_SN 10

namespace alignment
{
typedef struct
{
  double phi_min;
  double phi_max;
  double theta_min;
  double theta_max;
} angle_threshold_t;

static angle_threshold_t g_angle_thresholds[NUM_CAM_SN] = {
  { 0, 0, 0, 0 },       // 0, dummy
  { 30, 90, 60, 120 },  // 1
  {
      -30, 30, 60, 120,
  },                      // 2
  { -90, -30, 60, 120 },  // 3
  {
      65, 160, 98, 148,
  },  // 4 // x-aixs:95~1735 is correct
  {
      -80, 80, 90, 150,
  },                        // 5
  { -170, -30, 90, 150 },   // 6
  { 150, 180, 90, 120 },    // 7
  { -105, 50, 90, 150 },    // 8
  { -180, -150, 90, 120 },  // 9
};

static const double g_invR_T_values[NUM_CAM_SN][9] = {
  // 0: maps to nothing
  { 0 },
  // 1: 60 left LidarFront .
  { 0.87296, -0.026354, 0.48708, -0.486966, 0.01106, 0.873351, -0.028404, -0.999591, -0.003178 },
  // 2: 60 LidarFront .
  { -0.00243068, 0.0012012, 0.999996, -0.999964, -0.008128969999999999, -0.00242083, 0.008126029999999999, -0.999966,
    0.00122091 },
  // 3: 60 right LidarFront .
  { -0.87665, 0.0372601, 0.479683, -0.480141, -0.0039296, -0.877182, -0.030799, -0.999298, 0.021335 },
  // 4:
  { 0.925797, -0.175469, -0.334829, 0.0419238, -0.83262, 0.552256, -0.375689, -0.5253139999999999, -0.763481 },
  // 5:  120 LidarFront .port_b/cam_1 ....
  { -0.07728790000000001, -0.804028, 0.589547, -0.996919, 0.0543901, -0.0565157, 0.0133747, -0.592098, -0.805755 },
  // 6:
  { -0.8539099999999999, -0.220861, -0.47123, -0.023933, 0.921187, -0.388383, 0.5198700000000001, -0.320366,
    -0.791898 },
  // 7:  30 left_back LidarLeft ...
  { 0.143614, -0.131303, -0.980885, 0.0257818, -0.990326, 0.136342, -0.989298, -0.0448696, -0.13884 },
  // 8:  30 LidarFrontTop .
  { 0.008087874185622068, -0.03508299471642862, 0.9993516747236102, -0.9997309167020865, -0.02201223120156404,
    0.007318187448722202, 0.02174121618408908, -0.9991419544585523, -0.03525158662501271 },
  // 9:  30 righ_backt LidarRight ...
  { -0.276814, -0.256694, -0.926003, -0.028933, 0.965449, -0.25898, 0.960488, -0.044897, -0.274677 }
};

static const double g_invT_T_values[NUM_CAM_SN][3] = {
  { 0 },
  // 1:
  { -0.8388879671519999, -0.99186839427, -0.735884517888 },
  // 2
  { 0.17809259524218, -1.0594415615464, -0.5923675041245 },
  // 3
  { 0.804456976827, -1.1105255659342, -0.1471886111190001 },
  // 4
  { -0.2212733793831, 0.8425008786167, -0.8032340913773 },
  // 5
  { 0.3720845016199, 1.8580406653253, 3.9395786409569 },
  // 6
  { 0.8001775764700001, 1.638467632634, 0.577098819602 },
  // 7
  { -0.336396913412, 1.07235939354, -0.88623702788 },
  // 8
  { -0.1250909823227383, -0.3273729103370032, -0.4472947044387419 },
  // 9
  { 0.1589921658460001, 0.8971377608659999, -1.379370169824 },
};

static const double g_alignment_dist_coeff_values[NUM_CAM_SN][5] = {
  // 0: dummy
  { 0, 0, 0, 0, 0 },

  // 1
  { -1.3234694514851916e-01, -2.2877508585825934e+00, -2.6973306893876632e-03, 9.8102015044200194e-04,
    8.4113059659251626e+00 },

  // 2
  {
      -0.2801951, 0.0400105, -0.00253047, 0.00201554, 0.0851216,
  },

  // 3
  {
      -1.3234694514851916e-01, -2.2877508585825934e+00, -2.6973306893876632e-03, 9.8102015044200194e-04,
      8.4113059659251626e+00,
  },

  // 4
  {
      -0.384279, 0.188464, -0.00160702, 0.000320484, -0.048941,
  },

  // 5
  {
      -0.117567, -0.134832, 0.0100694, 0.00339625, 0.0829011,
  },

  // 6
  {
      -0.269525, -0.0169836, -0.00484475, 0.00881316, 0.0491113,
  },

  // 7
  {
      0.265471, -13.0192, -0.00588032, 0.00378908, 114.774,
  },

  // 8
  {
      -1.3234694514851916e-01, -2.2877508585825934e+00, -2.6973306893876632e-03, 9.8102015044200194e-04,
      8.4113059659251626e+00,
  },

  // 9
  {
      -0.206623, -0.511431, -0.00909592, 5.11685e-05, 6.7519,
  },

};

// 3*3 for each g_alig_camera_mat
static const double g_alignment_camera_mat_values[NUM_CAM_SN][9] = {
  // 0
  { 0, 0, 0, 0, 0, 0, 0, 0, 0 },

  // 1
  {
      1.8828750696345512e+03, 0, 9.7726620903028095e+02, 0, 1.8708459663727708e+03, 5.4527594054228973e+02, 0, 0, 1,
  },

  // 2
  {
      1864.08, 0, 958.784, 0, 1873.96, 604.981, 0, 0, 1,
  },

  // 3
  {
      1.8828750696345512e+03, 0, 9.7726620903028095e+02, 0, 1.8708459663727708e+03, 5.4527594054228973e+02, 0, 0, 1,
  },

  // 4
  {
      1006.37, 0, 947.756, 0, 1011.26, 615.522, 0, 0, 1,
  },

  // 5
  {
      1382.82, 0, 962.425, 0, 1327.62, 607.539, 0, 0, 1,
  },

  // 6
  {
      1270.75, 0, 957.399, 0, 1256.34, 614.28, 0, 0, 1,
  },

  // 7
  {
      3682.51, 0, 992.373, 0, 3707.93, 448.949, 0, 0, 1,
  },

  // 8
  {
      1.8828750696345512e+03, 0, 9.7726620903028095e+02, 0, 1.8708459663727708e+03, 5.4527594054228973e+02, 0, 0, 1,
  },

  // 9
  {
      3537.66, 0, 1043.23, 0, 3576.59, 419.304, 0, 0, 1,
  },

};

static cv::Mat g_invR_T[NUM_CAM_SN];
static cv::Mat g_invT_T[NUM_CAM_SN];
static cv::Mat g_alignment_camera_mat[NUM_CAM_SN];
static cv::Mat g_alignment_dist_coeff_mat[NUM_CAM_SN];

const cv::Mat& get_invR_T(const int cam_sn)
{
  assert(cam_sn >= 1);
  assert(cam_sn <= 9);
  if (g_invR_T[cam_sn].empty())
  {
    g_invR_T[cam_sn] = cv::Mat(3, 3, CV_64F, const_cast<double*>(g_invR_T_values[cam_sn]));
  }
  return g_invR_T[cam_sn];
}

const cv::Mat& get_invT_T(const int cam_sn)
{
  assert(cam_sn >= 1);
  assert(cam_sn <= 9);
  if (g_invT_T[cam_sn].empty())
  {
    g_invT_T[cam_sn] = cv::Mat(1, 3, CV_64F, const_cast<double*>(g_invT_T_values[cam_sn]));
  }
  return g_invT_T[cam_sn];
}

const cv::Mat& get_alignment_camera_mat(const int cam_sn)
{
  assert(cam_sn >= 1);
  assert(cam_sn <= 9);

  if (g_alignment_camera_mat[cam_sn].empty())
  {
    g_alignment_camera_mat[cam_sn] = cv::Mat(3, 3, CV_64F, const_cast<double*>(g_alignment_camera_mat_values[cam_sn]));
  }
  return g_alignment_camera_mat[cam_sn];
}

const cv::Mat& get_alignment_dist_coeff_mat(const int cam_sn)
{
  assert(cam_sn >= 1);
  assert(cam_sn <= 9);

  if (g_alignment_dist_coeff_mat[cam_sn].empty())
  {
    g_alignment_dist_coeff_mat[cam_sn] =
        cv::Mat(1, 5, CV_64F, const_cast<double*>(g_alignment_dist_coeff_values[cam_sn]));
  }
  return g_alignment_dist_coeff_mat[cam_sn];
}

double get_phi_min(const int cam_sn)
{
  assert(cam_sn >= 1);
  assert(cam_sn <= 9);
  return g_angle_thresholds[cam_sn].phi_min;
}

double get_phi_max(const int cam_sn)
{
  assert(cam_sn >= 1);
  assert(cam_sn <= 9);
  return g_angle_thresholds[cam_sn].phi_max;
}

double get_theta_min(const int cam_sn)
{
  assert(cam_sn >= 1);
  assert(cam_sn <= 9);
  return g_angle_thresholds[cam_sn].theta_min;
}

double get_theta_max(const int cam_sn)
{
  assert(cam_sn >= 1);
  assert(cam_sn <= 9);
  return g_angle_thresholds[cam_sn].theta_max;
}

std::string get_lidar_topic_by_cam_sn(const int cam_sn)
{
  // TODO: check topic name mapping
  std::string topics[10] = { "dummy",           // 0
                             "/LidarFront",     // 1
                             "/LidarFront",     // 2
                             "/LidarFront",     // 3
                             "/LidarLeft",      // 4
                             "/LidarFront",     // 5
                             "/LidarRight",     // 6
                             "/LidarLeft",      // 7
                             "/LidarFrontTop",  // 8
                             "/LidarRight" };   // 9
  assert(cam_sn >= 1);
  assert(cam_sn <= 9);
  return topics[cam_sn];
}
};  // namespace
