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
#include <mutex>

#include "drivenet/drivenet_group_b_b1_v2.h"

#include <msgs/DetectedObjectArray.h>

using namespace DriveNet;

/// camera layout
#if CAR_MODEL_IS_B1_V2
const std::vector<int> g_cam_ids{ camera::id::top_front_120, camera::id::right_front_60, camera::id::right_rear_60 };
#else
#error "car model is not well defined"
#endif

/// class
// DistanceEstimation g_dist_est;
Yolo_app g_yolo_app;
// CosmapGenerator g_cosmap_gener;

/// launch param
int g_car_id = 1;
int g_dist_est_mode = 0;
bool g_standard_FPS = 0;
bool g_display_flag = 0;
bool g_input_resize = 1;  // grabber input mode 0: 1920x1208, 1:608x384 yolo format
bool g_img_result_publish = 1;

/// function
void* run_yolo(void*);
void* run_interp(void*);
void* run_display(void*);

/// other param
bool g_is_infer_stop;
bool g_is_infer_data;
std::vector<bool> g_is_infer_datas(g_cam_ids.size());
bool g_is_compressed = false;

/// thread
pthread_mutex_t g_mtx_infer;
pthread_cond_t g_cnd_infer;
std::mutex g_cam_mutex;
std::mutex g_display_mutex;

/// ros publisher/subscriber
std::vector<ros::Publisher> g_bbox_pubs(g_cam_ids.size());
std::vector<image_transport::Publisher> g_img_pubs(g_cam_ids.size());
std::vector<msgs::DetectedObjectArray> g_doas;
// // grid map
// ros::Publisher g_occupancy_grid_publisher;

/// image
int g_img_w = camera::raw_image_width;
int g_img_h = camera::raw_image_height;
int g_rawimg_w = camera::raw_image_width;
int g_rawimg_h = camera::raw_image_height;
int g_img_size = g_img_w * g_img_h;
int g_rawimg_size = g_rawimg_w * g_rawimg_h;
std::vector<cv::Mat> g_mats(g_cam_ids.size());
std::vector<cv::Mat> g_mats_display(g_cam_ids.size());

/// object
std::vector<std::vector<ITRI_Bbox>> g_bboxs(g_cam_ids.size());
std::vector<std::vector<ITRI_Bbox>*> g_bbox_output(g_cam_ids.size());

/// detection information
std::vector<cv::Mat*> g_mat_srcs(g_cam_ids.size());
std::vector<uint32_t> g_mat_order(g_cam_ids.size());
std::vector<std_msgs::Header> g_headers(g_cam_ids.size());
std::vector<int> g_dist_rows(g_cam_ids.size());
std::vector<int> g_dist_cols(g_cam_ids.size());

// Prepare cv::Mat
void image_init()
{
  if (g_input_resize == 1)
  {
    g_img_w = camera::image_width;
    g_img_h = camera::image_height;
  }
  g_img_size = g_img_w * g_img_h;

  if (g_display_flag)
  {
    for (size_t ndx = 0; ndx < g_cam_ids.size(); ndx++)
    {
      g_mats_display[ndx] = cv::Mat(g_img_h, g_img_w, CV_8UC3, cv::Scalar(0));
    }
  }
  std_msgs::Header h;
  for (size_t ndx = 0; ndx < g_cam_ids.size(); ndx++)
  {
    g_is_infer_datas[ndx] = false;
    g_mats[ndx] = cv::Mat(g_img_h, g_img_w, CV_8UC3, cv::Scalar(0));
    g_mat_srcs[ndx] = &g_mats[ndx];
    g_mat_order[ndx] = ndx;
    g_headers[ndx] = h;
    g_bbox_output[ndx] = &g_bboxs[ndx];
    g_dist_cols[ndx] = g_rawimg_w;
    g_dist_rows[ndx] = g_rawimg_h;
  }
}

void sync_inference(int cam_order, std_msgs::Header& header, cv::Mat* mat, std::vector<ITRI_Bbox>* vbbx)
{
  pthread_mutex_lock(&g_mtx_infer);

  bool isPushData = false;
  if (!g_is_infer_datas[cam_order])
  {
    g_is_infer_datas[cam_order] = true;
    isPushData = true;

    if (isPushData)
    {
      g_headers[cam_order] = header;
      // std::cout << "Subscribe " <<  camera::topics[g_cam_ids[cam_order]] << " image." << std::endl;
    }
  }

  bool is_infer_data = true;
  for (size_t ndx = 0; ndx < g_cam_ids.size(); ndx++)
  {
    if (!g_is_infer_datas[ndx])
    {
      is_infer_data = false;
    }
  }
  if (is_infer_data)
  {
    g_is_infer_data = true;
    pthread_cond_signal(&g_cnd_infer);
  }
  pthread_mutex_unlock(&g_mtx_infer);

  while (g_is_infer_data)
    usleep(5);
}

void callback_cam_0(const sensor_msgs::Image::ConstPtr& msg)
{
  int cam_order = 0;
  if (!g_is_infer_datas[cam_order])
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    g_cam_mutex.lock();
    g_mats[cam_order] = cv_ptr->image;
    g_cam_mutex.unlock();
    std_msgs::Header h = msg->header;
    sync_inference(cam_order, h, &g_mats[cam_order], &g_bboxs[cam_order]);
  }
}

void callback_cam_1(const sensor_msgs::Image::ConstPtr& msg)
{
  int cam_order = 1;
  if (!g_is_infer_datas[cam_order])
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    g_cam_mutex.lock();
    g_mats[cam_order] = cv_ptr->image;
    g_cam_mutex.unlock();
    std_msgs::Header h = msg->header;
    sync_inference(cam_order, h, &g_mats[cam_order], &g_bboxs[cam_order]);
  }
}
void callback_cam_2(const sensor_msgs::Image::ConstPtr& msg)
{
  int cam_order = 2;
  if (!g_is_infer_datas[cam_order])
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    g_cam_mutex.lock();
    g_mats[cam_order] = cv_ptr->image;
    g_cam_mutex.unlock();
    std_msgs::Header h = msg->header;
    sync_inference(cam_order, h, &g_mats[cam_order], &g_bboxs[cam_order]);
  }
}

void callback_cam_0_decode(sensor_msgs::CompressedImage compressImg)
{
  int cam_order = 0;
  if (!g_is_infer_datas[cam_order])
  {
    g_cam_mutex.lock();
    cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(g_mats[cam_order]);
    g_cam_mutex.unlock();
    sync_inference(cam_order, compressImg.header, &g_mats[cam_order], &g_bboxs[cam_order]);
  }
}

void callback_cam_1_decode(sensor_msgs::CompressedImage compressImg)
{
  int cam_order = 1;
  if (!g_is_infer_datas[cam_order])
  {
    g_cam_mutex.lock();
    cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(g_mats[cam_order]);
    g_cam_mutex.unlock();
    sync_inference(cam_order, compressImg.header, &g_mats[cam_order], &g_bboxs[cam_order]);
  }
}
void callback_cam_2_decode(sensor_msgs::CompressedImage compressImg)
{
  int cam_order = 2;
  if (!g_is_infer_datas[cam_order])
  {
    g_cam_mutex.lock();
    cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(g_mats[cam_order]);
    g_cam_mutex.unlock();
    sync_inference(cam_order, compressImg.header, &g_mats[cam_order], &g_bboxs[cam_order]);
  }
}
void image_publisher(cv::Mat image, std_msgs::Header header, int cam_order)
{
  sensor_msgs::ImagePtr imgMsg;
  imgMsg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
  g_img_pubs[cam_order].publish(imgMsg);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "drivenet_group_b_b1_v2");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  g_is_infer_stop = false;
  g_is_infer_data = false;

  ros::param::get(ros::this_node::getName() + "/car_id", g_car_id);
  ros::param::get(ros::this_node::getName() + "/standard_fps", g_standard_FPS);
  ros::param::get(ros::this_node::getName() + "/display", g_display_flag);
  ros::param::get(ros::this_node::getName() + "/input_resize", g_input_resize);
  ros::param::get(ros::this_node::getName() + "/imgResult_publish", g_img_result_publish);
  ros::param::get(ros::this_node::getName() + "/dist_esti_mode", g_dist_est_mode);

  std::vector<std::string> cam_topicNames(g_cam_ids.size());
  std::vector<std::string> bbox_topicNames(g_cam_ids.size());
  std::vector<ros::Subscriber> cam_subs(g_cam_ids.size());
  static void (*f_cam_callbacks[])(const sensor_msgs::Image::ConstPtr&) = { callback_cam_0, callback_cam_1,
                                                                            callback_cam_2 };
  static void (*f_cam_decodes_callbacks[])(
      sensor_msgs::CompressedImage) = { callback_cam_0_decode, callback_cam_1_decode, callback_cam_2_decode };

  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    cam_topicNames[cam_order] = camera::topics[g_cam_ids[cam_order]];
    bbox_topicNames[cam_order] = camera::topics_obj[g_cam_ids[cam_order]];
    if (g_is_compressed)
    {
      cam_subs[cam_order] =
          nh.subscribe(cam_topicNames[cam_order] + std::string("/compressed"), 1, f_cam_decodes_callbacks[cam_order]);
    }
    else
    {
      cam_subs[cam_order] = nh.subscribe(cam_topicNames[cam_order], 1, f_cam_callbacks[cam_order]);
    }
    if (g_img_result_publish)
    {
      g_img_pubs[cam_order] = it.advertise(cam_topicNames[cam_order] + std::string("/detect_image"), 1);
    }
    g_bbox_pubs[cam_order] = nh.advertise<msgs::DetectedObjectArray>(bbox_topicNames[cam_order], 8);
  }

  // // occupancy grid map publisher
  // std::string occupancy_grid_topicName = camera::detect_result_occupancy_grid;
  // g_occupancy_grid_publisher = nh.advertise<nav_msgs::OccupancyGrid>(occupancy_grid_topicName, 1, true);

  pthread_mutex_init(&g_mtx_infer, NULL);
  pthread_cond_init(&g_cnd_infer, NULL);

  pthread_t thrdYolo, thrdInterp, thrdDisplay;
  pthread_create(&thrdYolo, NULL, &run_yolo, NULL);
  if (g_standard_FPS == 1)
    pthread_create(&thrdInterp, NULL, &run_interp, NULL);
  if (g_display_flag == 1)
    pthread_create(&thrdDisplay, NULL, &run_display, NULL);

  std::string pkg_path = ros::package::getPath("drivenet");
  std::string cfg_file = "/b1_v2_yolo_group_b.cfg";
  image_init();
  g_yolo_app.init_yolo(pkg_path, cfg_file);
  // g_dist_est.init(g_car_id, pkg_path, g_dist_est_mode);

  ros::MultiThreadedSpinner spinner(3);
  spinner.spin();

  g_is_infer_stop = true;
  pthread_join(thrdYolo, NULL);
  if (g_standard_FPS == 1)
    pthread_join(thrdInterp, NULL);
  if (g_display_flag == 1)
    pthread_join(thrdDisplay, NULL);

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
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      g_bbox_pubs[cam_order].publish(g_doas[cam_order]);
    }
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

  float distance = -1;
  detObj.distance = distance;

  camInfo.u = box.x1;
  camInfo.v = box.y1;
  camInfo.width = box.x2 - box.x1;
  camInfo.height = box.y2 - box.y1;
  camInfo.prob = box.prob;

  detObj.classId = translate_label(box.label);
  detObj.camInfo = camInfo;
  detObj.fusionSourceId = 0;

  return detObj;
}

void reset_data()
{
  g_is_infer_data = false;

  for (size_t ndx = 0; ndx < g_cam_ids.size(); ndx++)
  {
    g_is_infer_datas[ndx] = false;
    g_bboxs[ndx].clear();
  }
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
      pthread_cond_wait(&g_cnd_infer, &g_mtx_infer);
    pthread_mutex_unlock(&g_mtx_infer);

    // check data
    for (auto& mat : g_mat_srcs)
      isDataVaild &= CheckMatDataValid(*mat);
    if (!isDataVaild)
    {
      reset_data();
      isDataVaild = true;
      continue;
    }
    // copy data
    for (size_t ndx = 0; ndx < g_cam_ids.size(); ndx++)
    {
      g_cam_mutex.lock();
      matSrcsRaw_tmp[ndx] = g_mat_srcs[ndx]->clone();
      g_cam_mutex.unlock();
      matSrcs_tmp.push_back(&matSrcsRaw_tmp[ndx]);
    }

    headers_tmp = g_headers;
    vbbx_output_tmp = g_bbox_output;
    matOrder_tmp = g_mat_order;
    dist_cols_tmp = g_dist_cols;
    dist_rows_tmp = g_dist_rows;

    // reset data
    reset_data();

    // inference
    if (!g_input_resize)
      g_yolo_app.input_preprocess(matSrcs_tmp);
    else
      g_yolo_app.input_preprocess(matSrcs_tmp, g_input_resize, dist_cols_tmp, dist_rows_tmp);

    g_yolo_app.inference_yolo();
    g_yolo_app.get_yolo_result(&matOrder_tmp, vbbx_output_tmp);

    // publish results
    msgs::DetectedObjectArray doa;
    std::vector<msgs::DetectedObject> vDo;
    // grid map init
    // grid_map::GridMap costmap_ = g_cosmap_gener.initGridMap();

    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      std::vector<ITRI_Bbox>* tmpBBx = vbbx_output_tmp[cam_order];
      if (g_img_result_publish || g_display_flag)
      {
        if (!(*matSrcs_tmp[cam_order]).data)
        {
          std::cout << "Unable to read *matSrcs_tmp : id " << cam_order << "." << std::endl;
          continue;
        }
        else if ((*matSrcs_tmp[cam_order]).cols <= 0 || (*matSrcs_tmp[cam_order]).rows <= 0)
        {
          std::cout << "*matSrcs_tmp cols: " << (*matSrcs_tmp[cam_order]).cols
                    << ", rows: " << (*matSrcs_tmp[cam_order]).rows << std::endl;
          continue;
        }
        try
        {
          cv::resize((*matSrcs_tmp[cam_order]), M_display_tmp, cv::Size(g_rawimg_w, g_rawimg_h), 0, 0, 0);
        }
        catch (cv::Exception& e)
        {
          std::cout << "OpenCV Exception: " << std::endl << e.what() << std::endl;
          continue;
        }
        M_display = M_display_tmp;
      }

      msgs::DetectedObject detObj;
      std::vector<std::future<msgs::DetectedObject>> pool;
      for (auto const& box : *tmpBBx)
      {
        if (translate_label(box.label) == 0)
          continue;
        pool.push_back(std::async(std::launch::async, run_dist, box, cam_order));
        if (g_img_result_publish || g_display_flag)
        {
          class_color = get_label_color(box.label);
          cv::rectangle(M_display, cvPoint(box.x1, box.y1), cvPoint(box.x2, box.y2), class_color, 8);
        }
      }
      for (size_t i = 0; i < pool.size(); i++)
      {
        detObj = pool[i].get();
        vDo.push_back(detObj);
        // if (g_display_flag)
        // {
        //   if (detObj.bPoint.p0.x != 0 && detObj.bPoint.p0.z != 0)
        //   {
        //     int x1 = detObj.camInfo.u;
        //     int y1 = detObj.camInfo.v;
        //     float distance = detObj.distance;
        //     distance = truncateDecimalPrecision(distance, 1);
        //     std::string distance_str = floatToString_with_RealPrecision(distance);

        //     class_color = get_common_label_color(detObj.classId);
        //     cv::putText(M_display, distance_str + " m", cvPoint(x1 + 10, y1 - 10), 0, 1.5, class_color, 2);
        //   }
        // }
      }

      doa.header = headers_tmp[cam_order];
      doa.header.frame_id = "lidar";  // mapping to lidar coordinate
      doa.objects = vDo;
      // // object To grid map
      // costmap_[g_cosmap_gener.layer_name_] =
      //     g_cosmap_gener.makeCostmapFromObjects(costmap_, g_cosmap_gener.layer_name_, 8, doa, false);

      if (g_standard_FPS == 1)
      {
        g_doas[cam_order] = doa;
      }
      else
      {
        g_bbox_pubs[cam_order].publish(doa);
      }

      if (g_img_result_publish || g_display_flag)
      {
        g_display_mutex.lock();
        g_mats_display[cam_order] = M_display.clone();
        g_display_mutex.unlock();

        if (g_img_result_publish)
        {
          image_publisher(g_mats_display[cam_order], headers_tmp[cam_order], cam_order);
        }
      }
      vDo.clear();
      std::cout << "Detect " << camera::topics[g_cam_ids[cam_order]] << " image." << std::endl;
    }
    // grid map To Occpancy publisher
    // g_cosmap_gener.OccupancyMsgPublisher(costmap_, g_occupancy_grid_publisher, doa.header);

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

  std::vector<std::string> window_names(g_cam_ids.size());
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    window_names[cam_order] = camera::topics[g_cam_ids[cam_order]];
    cv::namedWindow(window_names[cam_order], cv::WINDOW_NORMAL);
    cv::resizeWindow(window_names[cam_order], 480, 360);
    cv::moveWindow(window_names[cam_order], 545 * cam_order, 30);
  }

  ros::Rate r(10);
  while (ros::ok() && !g_is_infer_stop)
  {
    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      if (g_mats_display[cam_order].data)
      {
        if (g_mats_display[cam_order].cols * g_mats_display[cam_order].rows == g_rawimg_size)
        {
          try
          {
            g_display_mutex.lock();
            cv::imshow(window_names[cam_order], g_mats_display[cam_order]);
            g_display_mutex.unlock();
            cv::waitKey(1);
          }
          catch (cv::Exception& e)
          {
            std::cout << "OpenCV Exception: " << std::endl << e.what() << std::endl;
          }
        }
      }
    }
    r.sleep();
  }

  std::cout << "run_display close" << std::endl;
  pthread_exit(0);
}
