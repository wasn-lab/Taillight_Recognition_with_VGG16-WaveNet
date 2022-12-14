#include <pthread.h>
#include <thread>
#include <future>
#include <mutex>
#include <std_msgs/Empty.h>

#include "drivenet/drivenet.h"

using namespace DriveNet;

// camera layout
#if CAR_MODEL_IS_B1_V2 || CAR_MODEL_IS_B1_V3 || CAR_MODEL_IS_C1 || CAR_MODEL_IS_C2 || CAR_MODEL_IS_C3
const std::vector<camera::id> g_cam_ids{ camera::id::front_top_close_120, camera::id::back_top_120 };
#else
#error "car model is not well defined"
#endif

// class
DistanceEstimation g_dist_est;
Yolo_app g_yolo_app;
// CosmapGenerator g_cosmap_gener;

// launch param
bool g_input_resize = true;  // mode 0: original resolution, 1: yolo resolution
bool g_standard_fps = false;
bool g_img_result_publish = true;
bool g_display_flag = false;
bool g_debug_flag = false;
int g_dist_est_mode = 0;

// function
void* run_interp(void* /*unused*/);
void* run_display(void* /*unused*/);
void* run_yolo(void* /*unused*/);

// other param
bool g_is_infer_stop;
bool g_is_infer_data;
std::vector<bool> g_is_infer_datas(g_cam_ids.size());

// thread
pthread_mutex_t g_mtx_infer;
pthread_cond_t g_cnd_infer;
std::mutex g_cam_mutex;
std::mutex g_display_mutex;

// ros publisher/subscriber
std::vector<ros::Publisher> g_bbox_pubs(g_cam_ids.size());
std::vector<ros::Publisher> g_heartbeat_pubs(g_cam_ids.size());
std::vector<ros::Publisher> g_cam_heartbeat_pubs(g_cam_ids.size());
std::vector<ros::Publisher> g_time_info_pubs(g_cam_ids.size());
std::vector<image_transport::Publisher> g_img_pubs(g_cam_ids.size());
std::vector<msgs::DetectedObjectArray> g_doas;
// // grid map
// ros::Publisher g_occupancy_grid_publisher;

// image
int g_img_w = camera::raw_image_width;
int g_img_h = camera::raw_image_height;
int g_rawimg_w = camera::raw_image_width;
int g_rawimg_h = camera::raw_image_height;
int g_img_size = g_img_w * g_img_h;
int g_rawimg_size = g_rawimg_w * g_rawimg_h;
std::vector<cv::Mat> g_mats(g_cam_ids.size());
std::vector<cv::Mat> g_mats_display(g_cam_ids.size());
std_msgs::Empty g_empty_msg;

// object
std::vector<std::vector<ITRI_Bbox>> g_bboxs(g_cam_ids.size());
std::vector<std::vector<ITRI_Bbox>*> g_bbox_output(g_cam_ids.size());

// detection information
std::vector<cv::Mat*> g_mat_srcs(g_cam_ids.size());
std::vector<uint32_t> g_mat_order(g_cam_ids.size());
std::vector<std_msgs::Header> g_headers(g_cam_ids.size());
std::vector<int> g_dist_rows(g_cam_ids.size());
std::vector<int> g_dist_cols(g_cam_ids.size());

// Prepare cv::Mat
void drivenet_init()
{
  if (g_input_resize)
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

  bool is_push_data = false;
  if (!g_is_infer_datas[cam_order])
  {
    g_is_infer_datas[cam_order] = true;
    is_push_data = true;

    if (is_push_data)
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
  {
    usleep(5);
  }
}

void callback_cam_front_top_close_120(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::front_top_close_120);
  int cam_order = std::distance(g_cam_ids.begin(), it);

  if (!g_is_infer_datas[cam_order])
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    g_cam_mutex.lock();
    g_mats[cam_order] = cv_ptr->image;
    g_cam_mutex.unlock();
    std_msgs::Header h = msg->header;
    sync_inference(cam_order, h, &g_mats[cam_order], &g_bboxs[cam_order]);
  }
  g_cam_heartbeat_pubs[cam_order].publish(g_empty_msg);
}
void callback_cam_back_top_120(const sensor_msgs::Image::ConstPtr& msg)
{
  auto it = std::find(g_cam_ids.begin(), g_cam_ids.end(), camera::id::back_top_120);
  int cam_order = std::distance(g_cam_ids.begin(), it);

  if (!g_is_infer_datas[cam_order])
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    g_cam_mutex.lock();
    g_mats[cam_order] = cv_ptr->image;
    g_cam_mutex.unlock();
    std_msgs::Header h = msg->header;
    sync_inference(cam_order, h, &g_mats[cam_order], &g_bboxs[cam_order]);
  }
  g_cam_heartbeat_pubs[cam_order].publish(g_empty_msg);
}

void detection_image_publisher(const cv::Mat& image, const std_msgs::Header& header, int cam_order)
{
  sensor_msgs::ImagePtr img_msg;
  img_msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
  g_img_pubs[cam_order].publish(img_msg);
}

void detection_image_heartbeat_publisher(int cam_order)
{
  std_msgs::Empty empty_msg;
  g_heartbeat_pubs[cam_order].publish(empty_msg);
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

void* run_interp(void* /*unused*/)
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
  pthread_exit(nullptr);
}

msgs::DetectedObject run_dist(ITRI_Bbox box, int cam_order)
{
  msgs::DetectedObject det_obj;
  std::vector<msgs::CamInfo> cam_info_vector;
  msgs::CamInfo cam_info;

  if (g_dist_est_mode)
  {
    msgs::BoxPoint box_point;
    // int l_check = 2;
    // int r_check = 2;
    float distance = -1;
    det_obj.distance = distance;
    /*
    if (g_cam_ids[cam_order] == camera::id::front_top_close_120)
    {
      l_check = g_dist_est.CheckPointInArea(g_dist_est.area[camera::id::front_top_close_120], box.x1, box.y2);
      r_check = g_dist_est.CheckPointInArea(g_dist_est.area[camera::id::front_top_close_120], box.x2, box.y2);
    }
    else if (g_cam_ids[cam_order] == camera::id::back_top_120)
    {
      l_check = g_dist_est.CheckPointInArea(g_dist_est.area[camera::id::back_top_120], box.x1, box.y2);
      r_check = g_dist_est.CheckPointInArea(g_dist_est.area[camera::id::back_top_120], box.x2, box.y2);
    }
    */

    // if (l_check == 0 && r_check == 0)
    // {
    box_point = g_dist_est.Get3dBBox(box.x1, box.y1, box.x2, box.y2, box.label, g_cam_ids[cam_order]);

    std::vector<float> left_point(2);
    std::vector<float> right_point(2);
    left_point[0] = box_point.p0.x;
    right_point[0] = box_point.p3.x;
    left_point[1] = box_point.p0.y;
    right_point[1] = box_point.p3.y;
    if (left_point[0] == 0 && left_point[1] == 0)
    {
      distance = -1;
    }
    else
    {
      distance = AbsoluteToRelativeDistance(left_point, right_point);  // relative distance
      det_obj.bPoint = box_point;
    }
    det_obj.distance = distance;
    // }
  }

  cam_info.u = box.x1;
  cam_info.v = box.y1;
  cam_info.width = box.x2 - box.x1;
  cam_info.height = box.y2 - box.y1;
  cam_info.prob = box.prob;
  cam_info.id = g_cam_ids[cam_order];

  cam_info_vector.push_back(cam_info);

  det_obj.classId = translate_label(box.label);
  det_obj.camInfo = cam_info_vector;
  det_obj.fusionSourceId = sensor_msgs_itri::FusionSourceId::Camera;

  return det_obj;
}

void* run_display(void* /*unused*/)
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
      if (g_mats_display[cam_order].data != nullptr)
      {
        if (g_mats_display[cam_order].cols * g_mats_display[cam_order].rows == g_img_size)
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
  pthread_exit(nullptr);
}

void* run_yolo(void* /*unused*/)
{
  std::cout << "run_inference start" << std::endl;
  std::vector<std_msgs::Header> headers_tmp;
  std::vector<std::vector<ITRI_Bbox>*> vbbx_output_tmp;
  std::vector<cv::Mat*> mat_srcs_tmp;
  std::vector<cv::Mat> mat_srcs_raw_tmp(g_cam_ids.size());
  std::vector<uint32_t> mat_order_tmp;
  std::vector<int> dist_cols_tmp;
  std::vector<int> dist_rows_tmp;

  std::vector<cv::Mat> m_display(g_cam_ids.size());
  cv::Scalar class_color;

  dist_cols_tmp = g_dist_cols;
  dist_rows_tmp = g_dist_rows;

  ros::Rate r(10);
  while (ros::ok() && !g_is_infer_stop)
  {
    bool is_data_vaild = true;

    // waiting for data
    pthread_mutex_lock(&g_mtx_infer);
    if (!g_is_infer_data)
    {
      pthread_cond_wait(&g_cnd_infer, &g_mtx_infer);
    }
    pthread_mutex_unlock(&g_mtx_infer);

    // check data
    for (auto& mat : g_mat_srcs)
    {
      is_data_vaild &= CheckMatDataValid(*mat);
    }
    if (!is_data_vaild)
    {
      reset_data();
      is_data_vaild = true;
      continue;
    }
    
    // copy data
    for (size_t ndx = 0; ndx < g_cam_ids.size(); ndx++)
    {
      g_cam_mutex.lock();
      mat_srcs_raw_tmp[ndx] = g_mat_srcs[ndx]->clone();
      g_cam_mutex.unlock();
      mat_srcs_tmp.push_back(&mat_srcs_raw_tmp[ndx]);
    }

    headers_tmp = g_headers;
    vbbx_output_tmp = g_bbox_output;
    mat_order_tmp = g_mat_order;

    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      std_msgs::Header header_msg;
      header_msg = headers_tmp[cam_order];
      g_time_info_pubs[cam_order].publish(header_msg);
    }

    // reset data
    reset_data();

    // inference
    if (!g_input_resize)
    {
      g_yolo_app.input_preprocess(mat_srcs_tmp);
    }
    else
    {
      g_yolo_app.input_preprocess(mat_srcs_tmp, static_cast<int>(g_input_resize), dist_cols_tmp, dist_rows_tmp);
    }
    g_yolo_app.inference_yolo();
    g_yolo_app.get_yolo_result(&mat_order_tmp, vbbx_output_tmp);

    // publish results
    std::vector<std::vector<msgs::DetectedObject>> v_do(g_cam_ids.size());
    // grid map init
    // grid_map::GridMap costmap_ = g_cosmap_gener.initGridMap();

    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      std::vector<ITRI_Bbox>* tmp_b_bx = vbbx_output_tmp[cam_order];

      if (g_img_result_publish || g_display_flag)
      {
        if ((*mat_srcs_tmp[cam_order]).data == nullptr)
        {
          std::cout << "Unable to read *matSrcs_tmp : id " << cam_order << "." << std::endl;
          continue;
        }
        else if ((*mat_srcs_tmp[cam_order]).cols <= 0 || (*mat_srcs_tmp[cam_order]).rows <= 0)
        {
          std::cout << "*matSrcs_tmp cols: " << (*mat_srcs_tmp[cam_order]).cols
                    << ", rows: " << (*mat_srcs_tmp[cam_order]).rows << std::endl;
          continue;
        }
        m_display[cam_order] = *mat_srcs_tmp[cam_order];
      }

      // tmp_b_bx = g_dist_est.MergeBbox(tmp_b_bx);
      msgs::DetectedObject det_obj;
      std::vector<std::future<msgs::DetectedObject>> pool;

      for (auto const& box : *tmp_b_bx)
      {
        if (translate_label(box.label) == 0)
        {
          continue;
        }
        det_obj = run_dist(box, cam_order);
        v_do[cam_order].push_back(det_obj);
        if (g_img_result_publish || g_display_flag)
        {
          class_color = get_label_color(box.label);
          PixelPosition position_1{ int(box.x1), int(box.y1) };
          PixelPosition position_2{ int(box.x2), int(box.y2) };
          if (g_input_resize)
          {
            transferPixelScaling(position_1);
            transferPixelScaling(position_2);
          }
          cv::rectangle(m_display[cam_order], cvPoint(position_1.u, position_1.v), cvPoint(position_2.u, position_2.v),
                        class_color, 3);
        }
        if (g_img_result_publish || g_display_flag)
        {
          if (g_debug_flag)
          {
            int x1 = det_obj.camInfo[0].u;
            int y1 = det_obj.camInfo[0].v;
            PixelPosition position_1{ x1, y1 };
            transferPixelScaling(position_1);

            float class_id = det_obj.camInfo[0].id;
            class_color = get_common_label_color(class_id);

            // draw class name
            std::string class_name = get_common_label_string(class_id);
            cv::putText(m_display[cam_order], class_name, cvPoint(position_1.u + 10, position_1.v + 10), 0, 0.3, class_color, 1);

            // draw distance
            if (det_obj.bPoint.p0.x != 0 && det_obj.bPoint.p0.z != 0)
            {
              float distance = det_obj.distance;
              distance = truncateDecimalPrecision(distance, 1);
              std::string distance_str = floatToString_with_RealPrecision(distance);
              cv::putText(m_display[cam_order], distance_str + " m", cvPoint(position_1.u + 5, position_1.v - 5), 0, 0.3,
                          class_color, 1);
            }
          }
        }

      }
    }

    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    { 
      msgs::DetectedObjectArray doa;
      doa.header = headers_tmp[cam_order];
      doa.header.frame_id = "lidar";  // mapping to lidar coordinate
      doa.objects = v_do[cam_order];

      // // object To grid map
      // costmap_[g_cosmap_gener.layer_name_] =
      //     g_cosmap_gener.makeCostmapFromObjects(costmap_, g_cosmap_gener.layer_name_, 8, doa, false);
      
      if (g_standard_fps)
      {
        g_doas[cam_order] = doa;
      }
      else
      {
        g_bbox_pubs[cam_order].publish(doa);
      }
      std::cout << "Detect " << camera::topics[g_cam_ids[cam_order]] << " image." << std::endl;
    }

    for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
    {
      if (g_display_flag)
      {
        g_display_mutex.lock();
        g_mats_display[cam_order] = m_display[cam_order].clone();
        g_display_mutex.unlock();
      }
      if (g_img_result_publish)
      {
        detection_image_publisher(m_display[cam_order], headers_tmp[cam_order], cam_order);
        detection_image_heartbeat_publisher(cam_order);
      }
      v_do[cam_order].clear();
    }
    // grid map To Occpancy publisher
    // g_cosmap_gener.OccupancyMsgPublisher(costmap_, g_occupancy_grid_publisher, doa.header);

    // reset data
    headers_tmp.clear();
    mat_srcs_tmp.clear();
    mat_order_tmp.clear();
    vbbx_output_tmp.clear();
    r.sleep();
  }
  std::cout << "run_inference close" << std::endl;
  pthread_exit(nullptr);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "drivenet_top");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  g_is_infer_stop = false;
  g_is_infer_data = false;

  // Get parameter in the launch file
  ros::param::get(ros::this_node::getName() + "/input_resize", g_input_resize);
  ros::param::get(ros::this_node::getName() + "/standard_fps", g_standard_fps);
  ros::param::get(ros::this_node::getName() + "/imgResult_publish", g_img_result_publish);
  ros::param::get(ros::this_node::getName() + "/display", g_display_flag);
  ros::param::get(ros::this_node::getName() + "/dist_esti_mode", g_dist_est_mode);

  // Setup subscriber and publisher
  std::vector<std::string> cam_topic_names(g_cam_ids.size());
  std::vector<std::string> bbox_topic_names(g_cam_ids.size());
  std::vector<ros::Subscriber> cam_subs(g_cam_ids.size());
  static void (*f_cam_callbacks[])(const sensor_msgs::Image::ConstPtr&) = { callback_cam_front_top_close_120,
                                                                            callback_cam_back_top_120 };

  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    cam_topic_names[cam_order] = camera::topics[g_cam_ids[cam_order]];
    bbox_topic_names[cam_order] = camera::topics_obj[g_cam_ids[cam_order]];

    cam_subs[cam_order] = nh.subscribe(cam_topic_names[cam_order], 1, f_cam_callbacks[cam_order]);

    if (g_img_result_publish)
    {
      g_img_pubs[cam_order] = it.advertise(cam_topic_names[cam_order] + std::string("/detect_image"), 1);
    }
    g_bbox_pubs[cam_order] = nh.advertise<msgs::DetectedObjectArray>(bbox_topic_names[cam_order], 8);
    g_heartbeat_pubs[cam_order] =
        nh.advertise<std_msgs::Empty>(cam_topic_names[cam_order] + std::string("/detect_image/heartbeat"), 1);
    g_cam_heartbeat_pubs[cam_order] =
        nh.advertise<std_msgs::Empty>(cam_topic_names[cam_order] + std::string("/drivenet_sub/heartbeat"), 1);
    g_time_info_pubs[cam_order] =
        nh.advertise<std_msgs::Header>(bbox_topic_names[cam_order] + std::string("/time_info"), 1);
  }

  // // occupancy grid map publisher
  // std::string occupancy_grid_topicName = camera::detect_result_occupancy_grid;
  // g_occupancy_grid_publisher = nh.advertise<nav_msgs::OccupancyGrid>(occupancy_grid_topicName, 1, true);

  // Initialize drivenet and other classes
  std::string pkg_path = ros::package::getPath("drivenet");
  std::string cfg_file = "/yolo_top.cfg";
  drivenet_init();
  g_yolo_app.init_yolo(pkg_path, cfg_file);
  g_dist_est.init(pkg_path, g_dist_est_mode);

  // Create the inference thread
  pthread_mutex_init(&g_mtx_infer, nullptr);
  pthread_cond_init(&g_cnd_infer, nullptr);
  pthread_t thrd_yolo, thrd_interp, thrd_display;
  pthread_create(&thrd_yolo, nullptr, &run_yolo, nullptr);
  if (g_standard_fps)
  {
    pthread_create(&thrd_interp, nullptr, &run_interp, nullptr);
  }
  if (g_display_flag)
  {
    pthread_create(&thrd_display, nullptr, &run_display, nullptr);
  }

  // Wait for input topic
  for (size_t cam_order = 0; cam_order < g_cam_ids.size(); cam_order++)
  {
    std::cout << "Wait for input topic " << cam_topic_names[cam_order] << std::endl;
    ros::topic::waitForMessage<sensor_msgs::Image>(cam_topic_names[cam_order]);
    std::cout << cam_topic_names[cam_order] << " is ready" << std::endl;
  }

  // Start to receive the message by the callback function
  ros::MultiThreadedSpinner spinner(int(g_cam_ids.size()));
  spinner.spin();

  // Close the program
  g_is_infer_stop = true;
  pthread_join(thrd_yolo, nullptr);
  if (g_standard_fps)
  {
    pthread_join(thrd_interp, nullptr);
  }
  if (g_display_flag)
  {
    pthread_join(thrd_display, nullptr);
  }

  pthread_mutex_destroy(&g_mtx_infer);
  g_yolo_app.delete_yolo_infer();
  ros::shutdown();

  return 0;
}