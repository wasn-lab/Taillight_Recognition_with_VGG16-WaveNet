#include "ros/ros.h"
#include "std_msgs/Header.h"
#include <ros/package.h>

#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <boost/thread.hpp>
#include <pthread.h>
#include <thread>
#include <future>

#include "drivenet/drivenet_120_1_b1.h"

#include <msgs/DetectedObjectArray.h>

using namespace DriveNet;

/// camera layout
#if CAR_MODEL_IS_B1
const std::vector<camera::id> g_cam_ids{
  camera::id::top_right_front_120, camera::id::top_right_rear_120, camera::id::top_left_front_120,
  camera::id::top_left_rear_120,
};
#else
#error "car model is not well defined"
#endif

/// class
DistanceEstimation g_dist_est;
Yolo_app g_yolo_app;
// CosmapGenerator g_cosmap_gener;

/// launch param
int g_car_id = 1;
int g_dist_est_mode = 0;
bool g_standard_fps = false;
bool g_display_flag = false;
bool g_input_resize = true;  // grabber input mode 0: 1920x1208, 1:608x384 yolo format
bool g_img_result_publish = true;

/// function
void* run_yolo(void*);
void* run_interp(void*);
void* run_display(void*);

/// other param
bool g_is_infer_stop;
bool g_is_infer_data;
bool g_is_infer_data_0;
bool g_is_infer_data_1;
bool g_is_infer_data_2;
bool g_is_infer_data_3;
bool g_is_compressed = false;
bool g_is_calibration = false;

/// thread
pthread_mutex_t g_mtx_infer;
pthread_cond_t g_cnd_infer;
std::mutex g_display_mutex;

/// ros publisher/subscriber
ros::Publisher g_pub120_0;
ros::Publisher g_pub120_1;
ros::Publisher g_pub120_2;
ros::Publisher g_pub120_3;
image_transport::Publisher g_pub_img_120_0;
image_transport::Publisher g_pub_img_120_1;
image_transport::Publisher g_pub_img_120_2;
image_transport::Publisher g_pub_img_120_3;
msgs::DetectedObjectArray g_doa120_0;
msgs::DetectedObjectArray g_doa120_1;
msgs::DetectedObjectArray g_doa120_2;
msgs::DetectedObjectArray g_doa120_3;
// grid map
// ros::Publisher g_occupancy_grid_publisher;

/// image
int g_rawimg_w = 1920;
int g_rawimg_h = 1208;
int g_img_w = 1920;
int g_img_h = 1208;
int g_img_size = g_img_w * g_img_h;
int g_rawimg_size = g_rawimg_w * g_rawimg_h;
cv::Mat g_mat120_0;
cv::Mat g_mat120_0_resize;
cv::Mat g_mat120_0_rect;
cv::Mat g_mat120_0_display;
cv::Mat g_mat120_1;
cv::Mat g_mat120_1_resize;
cv::Mat g_mat120_1_rect;
cv::Mat g_mat120_1_display;
cv::Mat g_mat120_2;
cv::Mat g_mat120_2_resize;
cv::Mat g_mat120_2_rect;
cv::Mat g_mat120_2_display;
cv::Mat g_mat120_3;
cv::Mat g_mat120_3_resize;
cv::Mat g_mat120_3_rect;
cv::Mat g_mat120_3_display;
cv::Mat g_camera_matrix, g_dist_coeffs;

/// object
std::vector<ITRI_Bbox> g_vbbx120_0;
std::vector<ITRI_Bbox> g_vbbx120_1;
std::vector<ITRI_Bbox> g_vbbx120_2;
std::vector<ITRI_Bbox> g_vbbx120_3;
std::vector<std::vector<ITRI_Bbox>*> g_vbbx_output;

/// detection information
std::vector<cv::Mat*> g_mat_srcs;
std::vector<uint32_t> g_mat_order;
std::vector<std_msgs::Header> g_headers;
std::vector<int> g_dist_rows;
std::vector<int> g_dist_cols;

// Prepare cv::Mat
void image_init()
{
  if (g_input_resize)
  {
    g_img_w = 608;
    g_img_h = 384;
  }
  g_img_size = g_img_w * g_img_h;

  if (g_display_flag)
  {
    g_mat120_0_display = cv::Mat(g_img_h, g_img_w, CV_8UC3, cv::Scalar(0));
    g_mat120_1_display = cv::Mat(g_img_h, g_img_w, CV_8UC3, cv::Scalar(0));
    g_mat120_2_display = cv::Mat(g_img_h, g_img_w, CV_8UC3, cv::Scalar(0));
    g_mat120_3_display = cv::Mat(g_img_h, g_img_w, CV_8UC3, cv::Scalar(0));
  }
}

void sync_inference(int cam_order, std_msgs::Header& header, cv::Mat* mat, std::vector<ITRI_Bbox>* vbbx, int dist_w,
                    int dist_h)
{
  pthread_mutex_lock(&g_mtx_infer);

  bool is_push_data = false;
  if (g_cam_ids[cam_order] == camera::id::top_right_front_120 && !g_is_infer_data_0)
  {
    g_is_infer_data_0 = true;
    is_push_data = true;
  }
  if (g_cam_ids[cam_order] == camera::id::top_right_rear_120 && !g_is_infer_data_1)
  {
    g_is_infer_data_1 = true;
    is_push_data = true;
  }
  if (g_cam_ids[cam_order] == camera::id::top_left_front_120 && !g_is_infer_data_2)
  {
    g_is_infer_data_2 = true;
    is_push_data = true;
  }
  if (g_cam_ids[cam_order] == camera::id::top_left_rear_120 && !g_is_infer_data_3)
  {
    g_is_infer_data_3 = true;
    is_push_data = true;
  }

  if (is_push_data)
  {
    g_mat_srcs.push_back(mat);
    g_mat_order.push_back(cam_order);
    g_vbbx_output.push_back(vbbx);
    g_headers.push_back(header);
    g_dist_cols.push_back(dist_w);
    g_dist_rows.push_back(dist_h);

    // std::cout << "Subscribe " <<  camera::topics[g_cam_ids[cam_order]] << " image." << std::endl;
  }

  if (g_mat_order.size() == g_cam_ids.size())
  {
    g_is_infer_data = true;
    pthread_cond_signal(&g_cnd_infer);
  }
  pthread_mutex_unlock(&g_mtx_infer);

  while (g_is_infer_data)
  {
    usleep(5);
  }
}

void callback_120_0(const sensor_msgs::Image::ConstPtr& msg)
{
  int cam_order = 0;
  if (!g_is_infer_data_0)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    g_mat120_0 = cv_ptr->image;
    std_msgs::Header h = msg->header;

    if (g_is_calibration)
    {
      if (g_input_resize)
      {
        cv::resize(g_mat120_0, g_mat120_0_resize, cv::Size(g_rawimg_w, g_rawimg_h));
        calibrationImage(g_mat120_0_resize, g_mat120_0_rect, g_camera_matrix, g_dist_coeffs);
      }
      else
      {
        calibrationImage(g_mat120_0, g_mat120_0_rect, g_camera_matrix, g_dist_coeffs);
      }
      sync_inference(cam_order, h, &g_mat120_0_rect, &g_vbbx120_0, 1920, 1208);
    }
    else
    {
      sync_inference(cam_order, h, &g_mat120_0, &g_vbbx120_0, 1920, 1208);
    }
  }
}

void callback_120_1(const sensor_msgs::Image::ConstPtr& msg)
{
  int cam_order = 1;
  if (!g_is_infer_data_1)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    g_mat120_1 = cv_ptr->image;
    std_msgs::Header h = msg->header;

    if (g_is_calibration)
    {
      if (g_input_resize)
      {
        cv::resize(g_mat120_1, g_mat120_1_resize, cv::Size(g_rawimg_w, g_rawimg_h));
        calibrationImage(g_mat120_1_resize, g_mat120_1_rect, g_camera_matrix, g_dist_coeffs);
      }
      else
      {
        calibrationImage(g_mat120_1, g_mat120_1_rect, g_camera_matrix, g_dist_coeffs);
      }
      sync_inference(cam_order, h, &g_mat120_1_rect, &g_vbbx120_1, 1920, 1208);
    }
    else
    {
      sync_inference(cam_order, h, &g_mat120_1, &g_vbbx120_1, 1920, 1208);
    }
  }
}

void callback_120_2(const sensor_msgs::Image::ConstPtr& msg)
{
  int cam_order = 2;
  if (!g_is_infer_data_2)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    g_mat120_2 = cv_ptr->image;
    std_msgs::Header h = msg->header;

    if (g_is_calibration)
    {
      if (g_input_resize)
      {
        cv::resize(g_mat120_2, g_mat120_2_resize, cv::Size(g_rawimg_w, g_rawimg_h));
        calibrationImage(g_mat120_2_resize, g_mat120_2_rect, g_camera_matrix, g_dist_coeffs);
      }
      else
      {
        calibrationImage(g_mat120_2, g_mat120_2_rect, g_camera_matrix, g_dist_coeffs);
      }
      sync_inference(cam_order, h, &g_mat120_2_rect, &g_vbbx120_2, 1920, 1208);
    }
    else
    {
      sync_inference(cam_order, h, &g_mat120_2, &g_vbbx120_2, 1920, 1208);
    }
  }
}

void callback_120_3(const sensor_msgs::Image::ConstPtr& msg)
{
  int cam_order = 3;
  if (!g_is_infer_data_3)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    g_mat120_3 = cv_ptr->image;
    std_msgs::Header h = msg->header;

    if (g_is_calibration)
    {
      if (g_input_resize)
      {
        cv::resize(g_mat120_3, g_mat120_3_resize, cv::Size(g_rawimg_w, g_rawimg_h));
        calibrationImage(g_mat120_3_resize, g_mat120_3_rect, g_camera_matrix, g_dist_coeffs);
      }
      else
      {
        calibrationImage(g_mat120_3, g_mat120_3_rect, g_camera_matrix, g_dist_coeffs);
      }
      sync_inference(cam_order, h, &g_mat120_3_rect, &g_vbbx120_3, 1920, 1208);
    }
    else
    {
      sync_inference(cam_order, h, &g_mat120_3, &g_vbbx120_3, 1920, 1208);
    }
  }
}

void callback_120_0_decode(sensor_msgs::CompressedImage compressImg)
{
  int cam_order = 0;
  if (!g_is_infer_data_0)
  {
    cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(g_mat120_0);
    std_msgs::Header h = compressImg.header;

    if (g_is_calibration)
    {
      if (g_input_resize)
      {
        cv::resize(g_mat120_0, g_mat120_0_resize, cv::Size(g_rawimg_w, g_rawimg_h));
        calibrationImage(g_mat120_0_resize, g_mat120_0_rect, g_camera_matrix, g_dist_coeffs);
      }
      else
      {
        calibrationImage(g_mat120_0, g_mat120_0_rect, g_camera_matrix, g_dist_coeffs);
      }
      sync_inference(cam_order, h, &g_mat120_0_rect, &g_vbbx120_0, 1920, 1208);
    }
    else
    {
      sync_inference(cam_order, h, &g_mat120_0, &g_vbbx120_0, 1920, 1208);
    }
  }
}

void callback_120_1_decode(sensor_msgs::CompressedImage compressImg)
{
  int cam_order = 1;
  if (!g_is_infer_data_1)
  {
    cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(g_mat120_1);
    std_msgs::Header h = compressImg.header;

    if (g_is_calibration)
    {
      if (g_input_resize)
      {
        cv::resize(g_mat120_1, g_mat120_1_resize, cv::Size(g_rawimg_w, g_rawimg_h));
        calibrationImage(g_mat120_1_resize, g_mat120_1_rect, g_camera_matrix, g_dist_coeffs);
      }
      else
      {
        calibrationImage(g_mat120_1, g_mat120_1_rect, g_camera_matrix, g_dist_coeffs);
      }
      sync_inference(cam_order, h, &g_mat120_1_rect, &g_vbbx120_1, 1920, 1208);
    }
    else
    {
      sync_inference(cam_order, h, &g_mat120_1, &g_vbbx120_1, 1920, 1208);
    }
  }
}

void callback_120_2_decode(sensor_msgs::CompressedImage compressImg)
{
  int cam_order = 2;
  if (!g_is_infer_data_2)
  {
    cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(g_mat120_2);
    std_msgs::Header h = compressImg.header;

    if (g_is_calibration)
    {
      if (g_input_resize)
      {
        cv::resize(g_mat120_2, g_mat120_2_resize, cv::Size(g_rawimg_w, g_rawimg_h));
        calibrationImage(g_mat120_2_resize, g_mat120_2_rect, g_camera_matrix, g_dist_coeffs);
      }
      else
      {
        calibrationImage(g_mat120_2, g_mat120_2_rect, g_camera_matrix, g_dist_coeffs);
      }
      sync_inference(cam_order, h, &g_mat120_2_rect, &g_vbbx120_2, 1920, 1208);
    }
    else
    {
      sync_inference(cam_order, h, &g_mat120_2, &g_vbbx120_2, 1920, 1208);
    }
  }
}

void callback_120_3_decode(sensor_msgs::CompressedImage compressImg)
{
  int cam_order = 3;
  if (!g_is_infer_data_3)
  {
    cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(g_mat120_3);
    std_msgs::Header h = compressImg.header;

    if (g_is_calibration)
    {
      if (g_input_resize)
      {
        cv::resize(g_mat120_3, g_mat120_3_resize, cv::Size(g_rawimg_w, g_rawimg_h));
        calibrationImage(g_mat120_3_resize, g_mat120_3_rect, g_camera_matrix, g_dist_coeffs);
      }
      else
      {
        calibrationImage(g_mat120_3, g_mat120_3_rect, g_camera_matrix, g_dist_coeffs);
      }
      sync_inference(cam_order, h, &g_mat120_3_rect, &g_vbbx120_3, 1920, 1208);
    }
    else
    {
      sync_inference(cam_order, h, &g_mat120_3, &g_vbbx120_3, 1920, 1208);
    }
  }
}

void image_publisher(const cv::Mat& image, const std_msgs::Header& header, int cam_order)
{
  sensor_msgs::ImagePtr imgMsg;
  imgMsg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();

  if (g_cam_ids[cam_order] == camera::id::top_right_front_120)
  {
    g_pub_img_120_0.publish(imgMsg);
  }
  else if (g_cam_ids[cam_order] == camera::id::top_right_rear_120)
  {
    g_pub_img_120_1.publish(imgMsg);
  }
  else if (g_cam_ids[cam_order] == camera::id::top_left_front_120)
  {
    g_pub_img_120_2.publish(imgMsg);
  }
  else if (g_cam_ids[cam_order] == camera::id::top_left_rear_120)
  {
    g_pub_img_120_3.publish(imgMsg);
  }
}

/// roslaunch drivenet drivenet120.launch
int main(int argc, char** argv)
{
  ros::init(argc, argv, "drivenet_120_1_b1");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  g_is_infer_stop = false;
  g_is_infer_data = false;

  ros::param::get(ros::this_node::getName() + "/car_id", g_car_id);
  ros::param::get(ros::this_node::getName() + "/standard_fps", g_standard_fps);
  ros::param::get(ros::this_node::getName() + "/display", g_display_flag);
  ros::param::get(ros::this_node::getName() + "/input_resize", g_input_resize);
  ros::param::get(ros::this_node::getName() + "/imgResult_publish", g_img_result_publish);
  ros::param::get(ros::this_node::getName() + "/dist_esti_mode", g_dist_est_mode);

  std::string cam120_0_topicName = camera::topics[camera::id::top_right_front_120];
  std::string cam120_1_topicName = camera::topics[camera::id::top_right_rear_120];
  std::string cam120_2_topicName = camera::topics[camera::id::top_left_front_120];
  std::string cam120_3_topicName = camera::topics[camera::id::top_left_rear_120];

  ros::Subscriber cam120_0, cam120_1, cam120_2, cam120_3;
  if (g_is_compressed)
  {
    cam120_0 = nh.subscribe(cam120_0_topicName + std::string("/compressed"), 1, callback_120_0_decode);
    cam120_1 = nh.subscribe(cam120_1_topicName + std::string("/compressed"), 1, callback_120_1_decode);
    cam120_2 = nh.subscribe(cam120_2_topicName + std::string("/compressed"), 1, callback_120_2_decode);
    cam120_3 = nh.subscribe(cam120_3_topicName + std::string("/compressed"), 1, callback_120_3_decode);
  }
  else
  {
    cam120_0 = nh.subscribe(cam120_0_topicName, 1, callback_120_0);
    cam120_1 = nh.subscribe(cam120_1_topicName, 1, callback_120_1);
    cam120_2 = nh.subscribe(cam120_2_topicName, 1, callback_120_2);
    cam120_3 = nh.subscribe(cam120_3_topicName, 1, callback_120_3);
  }

  if (g_img_result_publish)
  {
    g_pub_img_120_0 = it.advertise(cam120_0_topicName + std::string("/detect_image"), 1);
    g_pub_img_120_1 = it.advertise(cam120_1_topicName + std::string("/detect_image"), 1);
    g_pub_img_120_2 = it.advertise(cam120_2_topicName + std::string("/detect_image"), 1);
    g_pub_img_120_3 = it.advertise(cam120_3_topicName + std::string("/detect_image"), 1);
  }

  g_pub120_0 = nh.advertise<msgs::DetectedObjectArray>("/CamObjRightFront", 4);
  g_pub120_1 = nh.advertise<msgs::DetectedObjectArray>("/CamObjRightBack", 4);
  g_pub120_2 = nh.advertise<msgs::DetectedObjectArray>("/CamObjLeftFront", 4);
  g_pub120_3 = nh.advertise<msgs::DetectedObjectArray>("/CamObjLeftBack", 4);

  // // occupancy grid map publisher
  // std::string occupancy_grid_topicName = camera::detect_result_occupancy_grid;
  // g_occupancy_grid_publisher = nh.advertise<nav_msgs::OccupancyGrid>(occupancy_grid_topicName, 1, true);

  pthread_mutex_init(&g_mtx_infer, NULL);
  pthread_cond_init(&g_cnd_infer, NULL);

  pthread_t thrdYolo, thrdInterp, thrdDisplay;
  pthread_create(&thrdYolo, NULL, &run_yolo, NULL);
  if (g_standard_fps)
  {
    pthread_create(&thrdInterp, NULL, &run_interp, NULL);
  }
  if (g_display_flag)
  {
    pthread_create(&thrdDisplay, NULL, &run_display, NULL);
  }

  std::string pkg_path = ros::package::getPath("drivenet");
  std::string cfg_file = "/b1_yolo_120_1.cfg";
  image_init();
  g_yolo_app.init_yolo(pkg_path, cfg_file);
  g_dist_est.init(g_car_id, pkg_path, g_dist_est_mode);

  if (g_is_calibration)
  {
    cv::String calibMatrix_filepath = pkg_path + "/config/sf3324.yml";
    std::cout << "Calibration mode is open, calibMatrix filepath: " << calibMatrix_filepath << std::endl;
    loadCalibrationMatrix(calibMatrix_filepath, g_camera_matrix, g_dist_coeffs);
  }

  ros::MultiThreadedSpinner spinner(4);
  spinner.spin();

  g_is_infer_stop = true;
  pthread_join(thrdYolo, NULL);
  if (g_standard_fps)
  {
    pthread_join(thrdInterp, NULL);
  }
  if (g_display_flag)
  {
    pthread_join(thrdDisplay, NULL);
  }

  pthread_mutex_destroy(&g_mtx_infer);
  g_yolo_app.delete_yolo_infer();
  ros::shutdown();

  return 0;
}

void* run_interp(void*)
{
  std::cout << "run_interp start" << std::endl;
  ros::Rate r(30);
  while (ros::ok() && !g_is_infer_stop)
  {
    g_pub120_0.publish(g_doa120_0);
    g_pub120_1.publish(g_doa120_1);
    g_pub120_2.publish(g_doa120_2);
    g_pub120_3.publish(g_doa120_3);

    r.sleep();
  }
  std::cout << "run_interp close" << std::endl;
  pthread_exit(0);
}

msgs::DetectedObject run_dist(ITRI_Bbox box, int cam_order)
{
  msgs::DetectedObject detObj;
  msgs::BoxPoint boxPoint;
  msgs::CamInfo camInfo;

  int leftCheck = 2;
  int rightCheck = 2;
  float distance = -1;
  detObj.distance = distance;

  if (g_cam_ids[cam_order] == camera::id::top_right_front_120)
  {
    // Right front 120 range:
    leftCheck = 2;
    rightCheck = 2;
  }
  else if (g_cam_ids[cam_order] == camera::id::top_right_rear_120)
  {
    // Right back 120 range:
    // x axis: 0 ~ 7 meters
    // y axis: -9 ~ 6 meters
    // cv::Point LeftLinePoint1(127, 272);
    // cv::Point LeftLinePoint2(-1422, 1207);
    // cv::Point RightLinePoint1(1904, 272);
    // cv::Point RightLinePoint2(3548, 1207);
    leftCheck = 2;
    rightCheck = 2;
  }
  else if (g_cam_ids[cam_order] == camera::id::top_left_front_120)
  {
    // Left Front 120 range:
    leftCheck = 2;
    rightCheck = 2;
  }
  else if (g_cam_ids[cam_order] == camera::id::top_left_rear_120)
  {
    // Left back 120 range:
    leftCheck = 2;
    rightCheck = 2;
  }

  if (leftCheck == 0 && rightCheck == 0)
  {
    boxPoint = g_dist_est.Get3dBBox(box.x1, box.y1, box.x2, box.y2, box.label, g_cam_ids[cam_order]);

    std::vector<float> left_point(2);
    std::vector<float> right_point(2);
    if (g_cam_ids[cam_order] == camera::id::top_right_front_120 ||
        g_cam_ids[cam_order] == camera::id::top_right_rear_120)
    {
      left_point[0] = boxPoint.p4.x;
      right_point[0] = boxPoint.p0.x;
      left_point[1] = boxPoint.p4.y;
      right_point[1] = boxPoint.p0.y;
    }
    else if (g_cam_ids[cam_order] == camera::id::top_left_front_120 ||
             g_cam_ids[cam_order] == camera::id::top_left_rear_120)
    {
      left_point[0] = boxPoint.p3.x;
      right_point[0] = boxPoint.p7.x;
      left_point[1] = boxPoint.p3.y;
      right_point[1] = boxPoint.p7.y;
    }

    if (left_point[0] == 0 && left_point[1] == 0)
    {
      distance = -1;
    }
    else
    {
      distance = AbsoluteToRelativeDistance(left_point, right_point);  // relative distance
      detObj.bPoint = boxPoint;
    }
    detObj.distance = distance;
  }

  camInfo.u = box.x1;
  camInfo.v = box.y1;
  camInfo.width = box.x2 - box.x1;
  camInfo.height = box.y2 - box.y1;
  camInfo.prob = box.prob;
  camInfo.id = translate_label(box.label);

  detObj.classId = translate_label(box.label);
  detObj.camInfo = camInfo;
  detObj.fusionSourceId = sensor_msgs_itri::FusionSourceId::Camera;

  return detObj;
}

void reset_data()
{
  g_mat_srcs.clear();
  g_headers.clear();
  g_mat_order.clear();
  g_vbbx120_0.clear();
  g_vbbx120_1.clear();
  g_vbbx120_2.clear();
  g_vbbx120_3.clear();
  g_vbbx_output.clear();
  g_dist_cols.clear();
  g_dist_rows.clear();

  g_is_infer_data = false;
  g_is_infer_data_0 = false;
  g_is_infer_data_1 = false;
  g_is_infer_data_2 = false;
  g_is_infer_data_3 = false;
}

void* run_yolo(void*)
{
  std::cout << "run_inference start" << std::endl;
  std::vector<std_msgs::Header> headers_tmp;
  std::vector<std::vector<ITRI_Bbox>*> vbbx_output_tmp;
  std::vector<cv::Mat*> matSrcs_tmp;
  std::vector<cv::Mat> matSrcsRaw_tmp(g_cam_ids.size());
  std::vector<uint32_t> matOrder_tmp;
  std::vector<int> dist_cols_tmp;
  std::vector<int> dist_rows_tmp;

  cv::Mat M_display;
  cv::Mat M_display_tmp;
  cv::Scalar class_color;

  ros::Rate r(30);
  while (ros::ok() && !g_is_infer_stop)
  {
    bool isDataVaild = true;

    // waiting for data
    pthread_mutex_lock(&g_mtx_infer);
    if (!g_is_infer_data)
    {
      pthread_cond_wait(&g_cnd_infer, &g_mtx_infer);
    }
    pthread_mutex_unlock(&g_mtx_infer);

    // copy data
    for (size_t ndx = 0; ndx < g_cam_ids.size(); ndx++)
    {
      matSrcsRaw_tmp[ndx] = g_mat_srcs[ndx]->clone();
      matSrcs_tmp.push_back(&matSrcsRaw_tmp[ndx]);
    }
    headers_tmp = g_headers;
    vbbx_output_tmp = g_vbbx_output;
    matOrder_tmp = g_mat_order;
    dist_cols_tmp = g_dist_cols;
    dist_rows_tmp = g_dist_rows;

    // reset data
    reset_data();

    // check data
    for (auto& mat : g_mat_srcs)
    {
      isDataVaild &= CheckMatDataValid(*mat);
    }
    for (auto& mat : matSrcs_tmp)
    {
      isDataVaild &= CheckMatDataValid(*mat);
    }
    if (!isDataVaild)
    {
      reset_data();
      matSrcs_tmp.clear();
      isDataVaild = true;
      continue;
    }

    // inference
    if (!g_input_resize || g_is_calibration)
    {
      g_yolo_app.input_preprocess(matSrcs_tmp);
    }
    else
    {
      g_yolo_app.input_preprocess(matSrcs_tmp, g_input_resize, dist_cols_tmp, dist_rows_tmp);
    }

    g_yolo_app.inference_yolo();
    g_yolo_app.get_yolo_result(&matOrder_tmp, vbbx_output_tmp);

    // publish results
    msgs::DetectedObjectArray doa;
    std::vector<msgs::DetectedObject> vDo;
    // grid map init
    // grid_map::GridMap costmap_ = g_cosmap_gener.initGridMap();

    for (size_t ndx = 0; ndx < vbbx_output_tmp.size(); ndx++)
    {
      std::vector<ITRI_Bbox>* tmpBBx = vbbx_output_tmp[ndx];
      if (g_img_result_publish || g_display_flag)
      {
        if (!(*matSrcs_tmp[ndx]).data)
        {
          std::cout << "Unable to read *matSrcs_tmp : id " << ndx << "." << std::endl;
          continue;
        }
        else if ((*matSrcs_tmp[ndx]).cols <= 0 || (*matSrcs_tmp[ndx]).rows <= 0)
        {
          std::cout << "*matSrcs_tmp cols: " << (*matSrcs_tmp[ndx]).cols << ", rows: " << (*matSrcs_tmp[ndx]).rows
                    << std::endl;
          continue;
        }
        M_display = *matSrcs_tmp[ndx];
      }
      msgs::DetectedObject detObj;
      int cam_order = matOrder_tmp[ndx];
      std::vector<std::future<msgs::DetectedObject>> pool;
      for (auto const& box : *tmpBBx)
      {
        if (translate_label(box.label) == 0)
        {
          continue;
        }
        pool.push_back(std::async(std::launch::async, run_dist, box, cam_order));
        if (g_img_result_publish || g_display_flag)
        {
          class_color = get_label_color(box.label);
          PixelPosition position_1{ int(box.x1), int(box.y1) };
          PixelPosition position_2{ int(box.x2), int(box.y2) };
          transferPixelScaling(position_1);
          transferPixelScaling(position_2);
          cv::rectangle(M_display, cvPoint(position_1.u, position_1.v), cvPoint(position_2.u, position_2.v),
                        class_color, 3);
        }
      }
      for (size_t i = 0; i < pool.size(); i++)
      {
        detObj = pool[i].get();
        vDo.push_back(detObj);
        if (g_img_result_publish || g_display_flag)
        {
          if (detObj.bPoint.p0.x != 0 && detObj.bPoint.p0.z != 0)
          {
            PixelPosition position_1{ int(detObj.camInfo.u), int(detObj.camInfo.v) };
            transferPixelScaling(position_1);
            float distance = detObj.distance;
            distance = truncateDecimalPrecision(distance, 1);
            std::string distance_str = floatToString_with_RealPrecision(distance);

            class_color = get_common_label_color(detObj.classId);
            cv::putText(M_display, distance_str + " m", cvPoint(position_1.u + 10, position_1.v - 10), 0, 0.5, class_color, 2);
          }
        }
      }
      doa.header = headers_tmp[ndx];
      doa.header.frame_id = "lidar";
      doa.objects = vDo;

      // // object To grid map
      // costmap_[g_cosmap_gener.layer_name_] =
      //     g_cosmap_gener.makeCostmapFromObjects(costmap_, g_cosmap_gener.layer_name_, 8, doa, false);

      if (g_cam_ids[cam_order] == camera::id::top_right_front_120)
      {
        if (g_standard_fps)
        {
          g_doa120_0 = doa;
        }
        else
        {
          g_pub120_0.publish(doa);
        }

        if (g_img_result_publish || g_display_flag)
        {
          g_display_mutex.lock();
          g_mat120_0_display = M_display.clone();
          g_display_mutex.unlock();

          if (g_img_result_publish)
          {
            image_publisher(g_mat120_0_display, headers_tmp[ndx], cam_order);
          }
        }
      }
      else if (g_cam_ids[cam_order] == camera::id::top_right_rear_120)
      {
        if (g_standard_fps)
        {
          g_doa120_1 = doa;
        }
        else
        {
          g_pub120_1.publish(doa);
        }

        if (g_img_result_publish || g_display_flag)
        {
          g_display_mutex.lock();
          g_mat120_1_display = M_display.clone();
          g_display_mutex.unlock();

          if (g_img_result_publish)
          {
            image_publisher(g_mat120_1_display, headers_tmp[ndx], cam_order);
          }
        }
      }
      else if (g_cam_ids[cam_order] == camera::id::top_left_front_120)
      {
        if (g_standard_fps)
        {
          g_doa120_2 = doa;
        }
        else
        {
          g_pub120_2.publish(doa);
        }

        if (g_img_result_publish || g_display_flag)
        {
          g_display_mutex.lock();
          g_mat120_2_display = M_display.clone();
          g_display_mutex.unlock();

          if (g_img_result_publish)
          {
            image_publisher(g_mat120_2_display, headers_tmp[ndx], cam_order);
          }
        }
      }
      else if (g_cam_ids[cam_order] == camera::id::top_left_rear_120)
      {
        if (g_standard_fps)
        {
          g_doa120_3 = doa;
        }
        else
        {
          g_pub120_3.publish(doa);
        }

        if (g_img_result_publish || g_display_flag)
        {
          g_display_mutex.lock();
          g_mat120_3_display = M_display.clone();
          g_display_mutex.unlock();

          if (g_img_result_publish)
          {
            image_publisher(g_mat120_3_display, headers_tmp[ndx], cam_order);
          }
        }
      }
      vDo.clear();
    }
    // // grid map To Occpancy publisher
    // g_cosmap_gener.OccupancyMsgPublisher(costmap_, g_occupancy_grid_publisher, doa.header);

    std::cout << "Detect " << camera::topics[g_cam_ids[0]] << ", " << camera::topics[g_cam_ids[1]] << ", "
              << camera::topics[g_cam_ids[2]] << " and " << camera::topics[g_cam_ids[3]] << " image." << std::endl;

    // reset data
    headers_tmp.clear();
    matSrcs_tmp.clear();
    matOrder_tmp.clear();
    vbbx_output_tmp.clear();
    dist_cols_tmp.clear();
    dist_rows_tmp.clear();
    r.sleep();
  }

  std::cout << "run_inference close" << std::endl;
  pthread_exit(0);
}

void* run_display(void*)
{
  std::cout << "run_display start" << std::endl;
  cv::namedWindow("RightFront-120", CV_WINDOW_NORMAL);
  cv::namedWindow("RightBack-120", CV_WINDOW_NORMAL);
  cv::namedWindow("LeftFront-120", CV_WINDOW_NORMAL);
  cv::namedWindow("LeftBack-120", CV_WINDOW_NORMAL);
  cv::resizeWindow("RightFront-120", 480, 360);
  cv::resizeWindow("RightBack-120", 480, 360);
  cv::resizeWindow("LeftFront-120", 480, 360);
  cv::resizeWindow("LeftBack-120", 480, 360);
  cv::moveWindow("RightFront-120", 1025, 360);
  cv::moveWindow("RightBack-120", 1500, 360);
  cv::moveWindow("LeftFront-120", 545, 360);
  cv::moveWindow("LeftBack-120", 0, 360);

  // int marker_h_0, marker_h_1, marker_h_2, marker_h_3;
  // marker_h_0 = 590;
  // cv::Point boundaryMarker_0_1, boundaryMarker_0_2, boundaryMarker_0_3, boundaryMarker_0_4;
  // cv::Point boundaryMarker_1_1, boundaryMarker_1_2, boundaryMarker_1_3, boundaryMarker_1_4;
  // cv::Point boundaryMarker_2_1, boundaryMarker_2_2, boundaryMarker_2_3, boundaryMarker_2_4;
  // cv::Point boundaryMarker_3_1, boundaryMarker_3_2, boundaryMarker_3_3, boundaryMarker_3_4;
  // boundaryMarker(g_rawimg_w, boundaryMarker_0_1, boundaryMarker_0_2, boundaryMarker_0_3, boundaryMarker_0_4,
  // marker_h_0);
  // boundaryMarker(g_rawimg_w, boundaryMarker_1_1, boundaryMarker_1_2, boundaryMarker_1_3, boundaryMarker_1_4,
  // marker_h_1);
  // boundaryMarker(g_rawimg_w, boundaryMarker_2_1, boundaryMarker_2_2, boundaryMarker_2_3, boundaryMarker_2_4,
  // marker_h_2);
  // boundaryMarker(g_rawimg_w, boundaryMarker_3_1, boundaryMarker_3_2, boundaryMarker_3_3, boundaryMarker_3_4,
  // marker_h_3);

  ros::Rate r(10);
  while (ros::ok() && !g_is_infer_stop)
  {
    if (g_mat120_0_display.data || g_mat120_1_display.data || g_mat120_2_display.data || g_mat120_3_display.data)
    {
      if (g_mat120_0_display.cols * g_mat120_0_display.rows == g_img_size &&
          g_mat120_1_display.cols * g_mat120_1_display.rows == g_img_size &&
          g_mat120_2_display.cols * g_mat120_2_display.rows == g_img_size &&
          g_mat120_3_display.cols * g_mat120_3_display.rows == g_img_size)
      {
        try
        {
          g_display_mutex.lock();
          // cv::line(g_mat120_1_display, boundaryMarker_1_1, boundaryMarker_1_2, cv::Scalar(255, 255, 255), 1);
          // cv::line(g_mat120_1_display, boundaryMarker_1_3, boundaryMarker_1_4, cv::Scalar(255, 255, 255), 1);
          cv::imshow("RightFront-120", g_mat120_0_display);
          cv::imshow("RightBack-120", g_mat120_1_display);
          cv::imshow("LeftFront-120", g_mat120_2_display);
          cv::imshow("LeftBack-120", g_mat120_3_display);
          g_display_mutex.unlock();
          cv::waitKey(1);
        }
        catch (cv::Exception& e)
        {
          std::cout << "OpenCV Exception: " << std::endl << e.what() << std::endl;
        }
      }
    }
    r.sleep();
  }

  std::cout << "run_display close" << std::endl;
  pthread_exit(0);
}
