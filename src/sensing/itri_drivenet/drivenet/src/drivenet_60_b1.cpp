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

#include "drivenet/drivenet_60_b1.h"

#include <msgs/DetectedObjectArray.h>

using namespace DriveNet;

pthread_t thrdYolo;
pthread_t thrdInterp;
pthread_t thrdDisplay;
void* run_yolo(void*);
void* run_interp(void*);
void* run_display(void*);

bool isInferStop;
bool isInferData;
bool isInferData_0;
bool isInferData_1;
bool isInferData_2;
bool isCompressed = false;

pthread_mutex_t mtxInfer;
pthread_cond_t cndInfer;
std::mutex display_mutex;

std::string cam60_0_topicName;
std::string cam60_1_topicName;
std::string cam60_2_topicName;
ros::Subscriber cam60_0;
ros::Subscriber cam60_1;
ros::Subscriber cam60_2;
ros::Publisher pub60_0;
ros::Publisher pub60_1;
ros::Publisher pub60_2;
image_transport::Publisher pubImg_60_0;
image_transport::Publisher pubImg_60_1;
image_transport::Publisher pubImg_60_2;
sensor_msgs::ImagePtr imgMsg;
msgs::DetectedObjectArray doa60_0;
msgs::DetectedObjectArray doa60_1;
msgs::DetectedObjectArray doa60_2;

int img_w = 1920;
int img_h = 1208;
int rawimg_w = 1920;
int rawimg_h = 1208;
int img_size = img_w * img_h;
int rawimg_size = rawimg_w * rawimg_h;
cv::Mat mat60_0;
cv::Mat mat60_0_display;
std::vector<ITRI_Bbox> vBBX60_0;

cv::Mat mat60_1;
cv::Mat mat60_1_display;
std::vector<ITRI_Bbox> vBBX60_1;

cv::Mat mat60_2;
cv::Mat mat60_2_display;
std::vector<ITRI_Bbox> vBBX60_2;

std::vector<std::vector<ITRI_Bbox>*> vbbx_output;

std::vector<cv::Mat*> matSrcs;
std::vector<uint32_t> matOrder;
std::vector<std_msgs::Header> headers;
std::vector<int> dist_rows;
std::vector<int> dist_cols;

// Prepare cv::Mat
void image_init()
{
  if (input_resize == 1)
  {
    img_w = 608;
    img_h = 384;
  }
  img_size = img_w * img_h;

  if (display_flag)
  {
    mat60_0_display = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(0));
    mat60_1_display = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(0));
    mat60_2_display = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(0));
  }
}

void sync_inference(int cam_order, std_msgs::Header& header, cv::Mat* mat, std::vector<ITRI_Bbox>* vbbx, int dist_w,
                    int dist_h)
{
  pthread_mutex_lock(&mtxInfer);

  bool isPushData = false;
  if (cam_order == camera::id::right_60 && !isInferData_0)
  {
    isInferData_0 = true;
    isPushData = true;
  }
  if (cam_order == camera::id::front_60 && !isInferData_1)
  {
    isInferData_1 = true;
    isPushData = true;
  }
  if (cam_order == camera::id::left_60 && !isInferData_2)
  {
    isInferData_2 = true;
    isPushData = true;
  }

  if (isPushData)
  {
    matSrcs.push_back(mat);
    matOrder.push_back(cam_order);
    vbbx_output.push_back(vbbx);
    headers.push_back(header);
    dist_cols.push_back(dist_w);
    dist_rows.push_back(dist_h);

    // std::cout << "Subscribe " <<  camera::topics[cam_ids_[cam_order]] << " image." << std::endl;
  }

  if (matOrder.size() == cam_ids_.size())
  {
    isInferData = true;
    pthread_cond_signal(&cndInfer);
  }
  pthread_mutex_unlock(&mtxInfer);

  while (isInferData)
    usleep(5);
}

void callback_60_0(const sensor_msgs::Image::ConstPtr& msg)
{
  if (!isInferData_0)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    mat60_0 = cv_ptr->image;
    std_msgs::Header h = msg->header;
    sync_inference(camera::id::right_60, h, &mat60_0, &vBBX60_0, 1920, 1208);
  }
}

void callback_60_1(const sensor_msgs::Image::ConstPtr& msg)
{
  if (!isInferData_1)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    mat60_1 = cv_ptr->image;
    std_msgs::Header h = msg->header;
    sync_inference(camera::id::front_60, h, &mat60_1, &vBBX60_1, 1920, 1208);
  }
}

void callback_60_2(const sensor_msgs::Image::ConstPtr& msg)
{
  if (!isInferData_2)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    mat60_2 = cv_ptr->image;
    std_msgs::Header h = msg->header;
    sync_inference(camera::id::left_60, h, &mat60_2, &vBBX60_2, 1920, 1208);
  }
}

void callback_60_0_decode(sensor_msgs::CompressedImage compressImg)
{
  if (!isInferData_0)
  {
    cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(mat60_0);
    sync_inference(camera::id::right_60, compressImg.header, &mat60_0, &vBBX60_0, 1920, 1208);
  }
}

void callback_60_1_decode(sensor_msgs::CompressedImage compressImg)
{
  if (!isInferData_1)
  {
    cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(mat60_1);
    sync_inference(camera::id::front_60, compressImg.header, &mat60_1, &vBBX60_1, 1920, 1208);
  }
}

void callback_60_2_decode(sensor_msgs::CompressedImage compressImg)
{
  if (!isInferData_2)
  {
    cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(mat60_2);
    sync_inference(camera::id::left_60, compressImg.header, &mat60_2, &vBBX60_2, 1920, 1208);
  }
}

void image_publisher(cv::Mat image, std_msgs::Header header, int cam_order)
{
  imgMsg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();

  if (cam_order == camera::id::right_60)
    pubImg_60_0.publish(imgMsg);
  else if (cam_order == camera::id::front_60)
    pubImg_60_1.publish(imgMsg);
  else if (cam_order == camera::id::left_60)
    pubImg_60_2.publish(imgMsg);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "drivenet_60_b1");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  isInferStop = false;
  isInferData = false;

  ros::param::get(ros::this_node::getName() + "/car_id", car_id);
  ros::param::get(ros::this_node::getName() + "/standard_fps", standard_FPS);
  ros::param::get(ros::this_node::getName() + "/display", display_flag);
  ros::param::get(ros::this_node::getName() + "/input_resize", input_resize);
  ros::param::get(ros::this_node::getName() + "/imgResult_publish", imgResult_publish);

  cam60_0_topicName = camera::topics[camera::id::right_60];
  cam60_1_topicName = camera::topics[camera::id::front_60];
  cam60_2_topicName = camera::topics[camera::id::left_60];

  if (isCompressed)
  {
    cam60_0 = nh.subscribe(cam60_0_topicName + std::string("/compressed"), 1, callback_60_0_decode);
    cam60_1 = nh.subscribe(cam60_1_topicName + std::string("/compressed"), 1, callback_60_1_decode);
    cam60_2 = nh.subscribe(cam60_2_topicName + std::string("/compressed"), 1, callback_60_2_decode);
  }
  else
  {
    cam60_0 = nh.subscribe(cam60_0_topicName, 1, callback_60_0);
    cam60_1 = nh.subscribe(cam60_1_topicName, 1, callback_60_1);
    cam60_2 = nh.subscribe(cam60_2_topicName, 1, callback_60_2);
  }

  if (imgResult_publish)
  {
    pubImg_60_0 = it.advertise(cam60_0_topicName + std::string("/detect_image"), 1);
    pubImg_60_1 = it.advertise(cam60_1_topicName + std::string("/detect_image"), 1);
    pubImg_60_2 = it.advertise(cam60_2_topicName + std::string("/detect_image"), 1);
  }

  pub60_0 = nh.advertise<msgs::DetectedObjectArray>("/CamObjFrontRight", 8);
  pub60_1 = nh.advertise<msgs::DetectedObjectArray>("/CamObjFrontCenter", 8);
  pub60_2 = nh.advertise<msgs::DetectedObjectArray>("/CamObjFrontLeft", 8);

  pthread_mutex_init(&mtxInfer, NULL);
  pthread_cond_init(&cndInfer, NULL);

  pthread_create(&thrdYolo, NULL, &run_yolo, NULL);
  if (standard_FPS == 1)
    pthread_create(&thrdInterp, NULL, &run_interp, NULL);
  if (display_flag == 1)
    pthread_create(&thrdDisplay, NULL, &run_display, NULL);

  std::string pkg_path = ros::package::getPath("drivenet");
  std::string cfg_file = "/b1_yolo_60.cfg";
  image_init();
  yoloApp.init_yolo(pkg_path, cfg_file);
  distEst.init(car_id);

  ros::MultiThreadedSpinner spinner(3);
  spinner.spin();

  isInferStop = true;
  pthread_join(thrdYolo, NULL);
  if (standard_FPS == 1)
    pthread_join(thrdInterp, NULL);
  if (display_flag == 1)
    pthread_join(thrdDisplay, NULL);

  pthread_mutex_destroy(&mtxInfer);
  yoloApp.delete_yolo_infer();
  ros::shutdown();

  return 0;
}

void* run_interp(void*)
{
  std::cout << "run_interp start" << std::endl;
  ros::Rate r(30);
  while (ros::ok() && !isInferStop)
  {
    pub60_0.publish(doa60_0);
    pub60_1.publish(doa60_1);
    pub60_2.publish(doa60_2);
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

  bool BoxPass_flag = false;

  if (cam_order == camera::id::right_60)
  {
    // Front right 60 range:
    // x axis: 1 - 10 meters
    // y axis: -5 ~ -30 meters

    BoxPass_flag = checkBoxInArea(distEst.camFR60_area, box.x1, box.y2, box.x2, box.y2);
  }
  else if (cam_order == camera::id::front_60)
  {
    // Front center 60 range:
    // x axis: 7 ~ 50 meters
    // y axis: -10 ~ 10 meters

    BoxPass_flag = checkBoxInArea(distEst.camFC60_area, box.x1, box.y2, box.x2, box.y2);
  }
  else if (cam_order == camera::id::left_60)
  {
    // Front left 60 range:
    // x axis: 0 - 10 meters
    // y axis: 4 ~ 30 meters

    BoxPass_flag = checkBoxInArea(distEst.camFL60_area, box.x1, box.y2, box.x2, box.y2);
    // BoxPass_flag = false;
  }

  if (BoxPass_flag)
  {
    boxPoint = distEst.Get3dBBox(box.x1, box.y1, box.x2, box.y2, box.label, cam_order);
    detObj.bPoint = boxPoint;
  }

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
  matSrcs.clear();
  headers.clear();
  matOrder.clear();
  vBBX60_0.clear();
  vBBX60_1.clear();
  vBBX60_2.clear();
  vbbx_output.clear();
  dist_cols.clear();
  dist_rows.clear();

  isInferData = false;
  isInferData_0 = false;
  isInferData_1 = false;
  isInferData_2 = false;
}

void* run_yolo(void*)
{
  std::cout << "run_inference start" << std::endl;
  std::vector<std_msgs::Header> headers_tmp;
  std::vector<std::vector<ITRI_Bbox>*> vbbx_output_tmp;
  std::vector<cv::Mat*> matSrcs_tmp;
  std::vector<cv::Mat> matSrcsRaw_tmp(cam_ids_.size());
  std::vector<uint32_t> matOrder_tmp;
  std::vector<int> dist_cols_tmp;
  std::vector<int> dist_rows_tmp;

  cv::Mat M_display;
  cv::Mat M_display_tmp;
  std::vector<cv::Scalar> cls_color = { cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0),
                                        cv::Scalar(125, 125, 125) };
  cv::Scalar class_color;

  ros::Rate r(30);
  while (ros::ok() && !isInferStop)
  {
    bool isDataVaild = true;

    // waiting for data
    pthread_mutex_lock(&mtxInfer);
    if (!isInferData)
      pthread_cond_wait(&cndInfer, &mtxInfer);
    pthread_mutex_unlock(&mtxInfer);

    // copy data
    for (size_t ndx = 0; ndx < cam_ids_.size(); ndx++)
    {
      matSrcsRaw_tmp[ndx] = matSrcs[ndx]->clone();
      matSrcs_tmp.push_back(&matSrcsRaw_tmp[ndx]);
    }

    headers_tmp = headers;
    vbbx_output_tmp = vbbx_output;
    matOrder_tmp = matOrder;
    dist_cols_tmp = dist_cols;
    dist_rows_tmp = dist_rows;

    // reset data
    reset_data();

    // check data
    for (auto& mat : matSrcs)
      isDataVaild &= CheckMatDataValid(*mat);
    for (auto& mat : matSrcs_tmp)
      isDataVaild &= CheckMatDataValid(*mat);
    if (!isDataVaild)
    {
      reset_data();
      matSrcs_tmp.clear();
      isDataVaild = true;
      continue;
    }

    // inference
    if (!input_resize)
      yoloApp.input_preprocess(matSrcs_tmp);
    else
      yoloApp.input_preprocess(matSrcs_tmp, input_resize, dist_cols_tmp, dist_rows_tmp);

    yoloApp.inference_yolo();
    yoloApp.get_yolo_result(&matOrder_tmp, vbbx_output_tmp);

    // publish results
    msgs::DetectedObjectArray doa;
    std::vector<msgs::DetectedObject> vDo;
    for (size_t ndx = 0; ndx < vbbx_output_tmp.size(); ndx++)
    {
      std::vector<ITRI_Bbox>* tmpBBx = vbbx_output_tmp[ndx];
      if (imgResult_publish || display_flag)
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
        try
        {
          cv::resize((*matSrcs_tmp[ndx]), M_display_tmp, cv::Size(rawimg_w, rawimg_h), 0, 0, 0);
        }
        catch (cv::Exception& e)
        {
          std::cout << "OpenCV Exception: " << std::endl << e.what() << std::endl;
          continue;
        }
        M_display = M_display_tmp;
      }

      msgs::DetectedObject detObj;
      int cam_order = matOrder_tmp[ndx];
      std::vector<std::future<msgs::DetectedObject>> pool;
      for (auto const& box : *tmpBBx)
      {
        if (translate_label(box.label) == 0)
          continue;
        pool.push_back(std::async(std::launch::async, run_dist, box, cam_order));
        if (imgResult_publish || display_flag)
        {
          class_color = get_labelColor(cls_color, box.label);
          cv::rectangle(M_display, cvPoint(box.x1, box.y1), cvPoint(box.x2, box.y2), class_color, 8);
        }
      }
      for (size_t i = 0; i < pool.size(); i++)
      {
        detObj = pool[i].get();
        vDo.push_back(detObj);
        if (display_flag)
        {
          if (detObj.bPoint.p0.x != 0 && detObj.bPoint.p0.z != 0)
          {
            int x1 = detObj.camInfo.u;
            int y1 = detObj.camInfo.v;
            float distMeter_p0x = detObj.bPoint.p0.x;
            // float distMeter_p3x = detObj.bPoint.p3.x;
            // float distMeter_p0y = detObj.bPoint.p0.y;
            // float distMeter_p3y = detObj.bPoint.p3.y;

            // float centerPoint[2];
            // centerPoint[0] = (distMeter_p0x + distMeter_p3x) / 2;
            // centerPoint[1] = (distMeter_p0y + distMeter_p3y) / 2;
            // float distance = sqrt(pow(centerPoint[0], 2) + pow(centerPoint[1], 2)); //relative distance
            float distance = distMeter_p0x; //vertical distance
            distance = truncateDecimalPrecision(distance, 1);
            std::string distance_str = floatToString_with_RealPrecision(distance);

            class_color = get_commonLabelColor(cls_color, detObj.classId);
            cv::putText(M_display, distance_str + " m", cvPoint(x1 + 10, y1 - 10), 0, 1.5, class_color, 2);
          }
        }
      }

      doa.header = headers_tmp[ndx];
      doa.header.frame_id = "lidar";  // mapping to lidar coordinate
      doa.objects = vDo;

      if (cam_order == camera::id::right_60)
      {
        if (standard_FPS == 1)
          doa60_0 = doa;
        else
          pub60_0.publish(doa);

        if (imgResult_publish || display_flag)
        {
          display_mutex.lock();
          mat60_0_display = M_display.clone();
          display_mutex.unlock();

          if (imgResult_publish)
          {
            image_publisher(mat60_0_display, headers_tmp[ndx], cam_order);
          }
        }
      }
      else if (cam_order == camera::id::front_60)
      {
        if (standard_FPS == 1)
          doa60_1 = doa;
        else
          pub60_1.publish(doa);

        if (imgResult_publish || display_flag)
        {
          display_mutex.lock();
          mat60_1_display = M_display.clone();
          display_mutex.unlock();

          if (imgResult_publish)
          {
            image_publisher(mat60_1_display, headers_tmp[ndx], cam_order);
          }
        }
      }
      else if (cam_order == camera::id::left_60)
      {
        if (standard_FPS == 1)
          doa60_2 = doa;
        else
          pub60_2.publish(doa);

        if (imgResult_publish || display_flag)
        {
          display_mutex.lock();
          mat60_2_display = M_display.clone();
          display_mutex.unlock();

          if (imgResult_publish)
          {
            image_publisher(mat60_2_display, headers_tmp[ndx], cam_order);
          }
        }
      }
      vDo.clear();
    }

    std::cout << "Detect " << camera::topics[cam_ids_[0]] << ", " << camera::topics[cam_ids_[1]] << " and "
              << camera::topics[cam_ids_[2]] << " image." << std::endl;

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
  cv::namedWindow("RightSide-60", cv::WINDOW_NORMAL);
  cv::namedWindow("Center-60", cv::WINDOW_NORMAL);
  cv::namedWindow("LeftSide-60", cv::WINDOW_NORMAL);
  cv::resizeWindow("RightSide-60", 480, 360);
  cv::resizeWindow("Center-60", 480, 360);
  cv::resizeWindow("LeftSide-60", 480, 360);
  cv::moveWindow("RightSide-60", 1025, 30);
  cv::moveWindow("Center-60", 545, 30);
  cv::moveWindow("LeftSide-60", 0, 30);

  int marker_h_1;  //, marker_h_0, marker_h_2;
  marker_h_1 = 914;

  // cv::Point boundaryMarker_0_1, boundaryMarker_0_2, boundaryMarker_0_3, boundaryMarker_0_4;
  cv::Point boundaryMarker_1_1, boundaryMarker_1_2, boundaryMarker_1_3, boundaryMarker_1_4;
  // cv::Point boundaryMarker_2_1, boundaryMarker_2_2, boundaryMarker_2_3, boundaryMarker_2_4;
  // boundaryMarker(rawimg_w, boundaryMarker_0_1, boundaryMarker_0_2, boundaryMarker_0_3, boundaryMarker_0_4,
  // marker_h_0);
  boundaryMarker(rawimg_w, boundaryMarker_1_1, boundaryMarker_1_2, boundaryMarker_1_3, boundaryMarker_1_4, marker_h_1);
  // boundaryMarker(rawimg_w, boundaryMarker_2_1, boundaryMarker_2_2, boundaryMarker_2_3, boundaryMarker_2_4,
  // marker_h_2);

  ros::Rate r(10);
  while (ros::ok() && !isInferStop)
  {
    if (mat60_0_display.data || mat60_1_display.data || mat60_2_display.data)
    {
      if (mat60_0_display.cols * mat60_0_display.rows == rawimg_size &&
          mat60_1_display.cols * mat60_1_display.rows == rawimg_size &&
          mat60_2_display.cols * mat60_2_display.rows == rawimg_size)
      {
        try
        {
          display_mutex.lock();
          cv::line(mat60_1_display, boundaryMarker_1_1, boundaryMarker_1_2, cv::Scalar(255, 255, 255), 1);
          cv::line(mat60_1_display, boundaryMarker_1_3, boundaryMarker_1_4, cv::Scalar(255, 255, 255), 1);
          cv::imshow("RightSide-60", mat60_0_display);
          cv::imshow("Center-60", mat60_1_display);
          cv::imshow("LeftSide-60", mat60_2_display);
          display_mutex.unlock();
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
