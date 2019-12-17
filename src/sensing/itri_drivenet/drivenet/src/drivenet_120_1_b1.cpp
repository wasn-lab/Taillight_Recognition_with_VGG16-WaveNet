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
bool isInferData_3;
bool isCompressed = false;
bool isCalibration = false;

pthread_mutex_t mtxInfer;
pthread_cond_t cndInfer;
std::mutex display_mutex;

std::string cam120_0_topicName;
std::string cam120_1_topicName;
std::string cam120_2_topicName;
std::string cam120_3_topicName;
ros::Subscriber cam120_0;
ros::Subscriber cam120_1;
ros::Subscriber cam120_2;
ros::Subscriber cam120_3;
ros::Publisher pub120_0;
ros::Publisher pub120_1;
ros::Publisher pub120_2;
ros::Publisher pub120_3;
image_transport::Publisher pubImg_120_0;
image_transport::Publisher pubImg_120_1;
image_transport::Publisher pubImg_120_2;
image_transport::Publisher pubImg_120_3;
sensor_msgs::ImagePtr imgMsg;
msgs::DetectedObjectArray doa120_0;
msgs::DetectedObjectArray doa120_1;
msgs::DetectedObjectArray doa120_2;
msgs::DetectedObjectArray doa120_3;

int rawimg_w = 1920;
int rawimg_h = 1208;
int img_w = 1920;
int img_h = 1208;
int img_size = img_w * img_h;
int rawimg_size = rawimg_w * rawimg_h;

cv::Mat mat120_0;
cv::Mat mat120_0_resize;
cv::Mat mat120_0_rect;
cv::Mat mat120_0_display;
std::vector<ITRI_Bbox> vBBX120_0;

cv::Mat mat120_1;
cv::Mat mat120_1_resize;
cv::Mat mat120_1_rect;
cv::Mat mat120_1_display;
std::vector<ITRI_Bbox> vBBX120_1;

cv::Mat mat120_2;
cv::Mat mat120_2_resize;
cv::Mat mat120_2_rect;
cv::Mat mat120_2_display;
std::vector<ITRI_Bbox> vBBX120_2;

cv::Mat mat120_3;
cv::Mat mat120_3_resize;
cv::Mat mat120_3_rect;
cv::Mat mat120_3_display;
std::vector<ITRI_Bbox> vBBX120_3;

std::vector<std::vector<ITRI_Bbox>*> vbbx_output;

cv::Mat cameraMatrix, distCoeffs;

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
    mat120_0_display = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(0));
    mat120_1_display = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(0));
    mat120_2_display = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(0));
    mat120_3_display = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(0));
  }
}

void sync_inference(int cam_order, std_msgs::Header& header, cv::Mat* mat, std::vector<ITRI_Bbox>* vbbx, int dist_w,
                    int dist_h)
{
  pthread_mutex_lock(&mtxInfer);

  bool isPushData = false;
  if (cam_order == camera::id::top_right_front_120 && !isInferData_0)
  {
    isInferData_0 = true;
    isPushData = true;
  }
  if (cam_order == camera::id::top_right_rear_120 && !isInferData_1)
  {
    isInferData_1 = true;
    isPushData = true;
  }
  if (cam_order == camera::id::top_left_front_120 && !isInferData_2)
  {
    isInferData_2 = true;
    isPushData = true;
  }
  if (cam_order == camera::id::top_left_rear_120 && !isInferData_3)
  {
    isInferData_3 = true;
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

void callback_120_0(const sensor_msgs::Image::ConstPtr& msg)
{
  if (!isInferData_0)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    mat120_0 = cv_ptr->image;
    std_msgs::Header h = msg->header;

    if (isCalibration)
    {
      if (input_resize)
      {
        cv::resize(mat120_0, mat120_0_resize, cv::Size(rawimg_w, rawimg_h));
        calibrationImage(mat120_0_resize, mat120_0_rect, cameraMatrix, distCoeffs);
      }
      else
      {
        calibrationImage(mat120_0, mat120_0_rect, cameraMatrix, distCoeffs);
      }
      sync_inference(camera::id::top_right_front_120, h, &mat120_0_rect, &vBBX120_0, 1920, 1208);
    }
    else
    {
      sync_inference(camera::id::top_right_front_120, h, &mat120_0, &vBBX120_0, 1920, 1208);
    }
  }
}

void callback_120_1(const sensor_msgs::Image::ConstPtr& msg)
{
  if (!isInferData_1)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    mat120_1 = cv_ptr->image;
    std_msgs::Header h = msg->header;

    if (isCalibration)
    {
      if (input_resize)
      {
        cv::resize(mat120_1, mat120_1_resize, cv::Size(rawimg_w, rawimg_h));
        calibrationImage(mat120_1_resize, mat120_1_rect, cameraMatrix, distCoeffs);
      }
      else
      {
        calibrationImage(mat120_1, mat120_1_rect, cameraMatrix, distCoeffs);
      }
      sync_inference(camera::id::top_right_rear_120, h, &mat120_1_rect, &vBBX120_1, 1920, 1208);
    }
    else
    {
      sync_inference(camera::id::top_right_rear_120, h, &mat120_1, &vBBX120_1, 1920, 1208);
    }
  }
}

void callback_120_2(const sensor_msgs::Image::ConstPtr& msg)
{
  if (!isInferData_2)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    mat120_2 = cv_ptr->image;
    std_msgs::Header h = msg->header;

    if (isCalibration)
    {
      if (input_resize)
      {
        cv::resize(mat120_2, mat120_2_resize, cv::Size(rawimg_w, rawimg_h));
        calibrationImage(mat120_2_resize, mat120_2_rect, cameraMatrix, distCoeffs);
      }
      else
      {
        calibrationImage(mat120_2, mat120_2_rect, cameraMatrix, distCoeffs);
      }
      sync_inference(camera::id::top_left_front_120, h, &mat120_2_rect, &vBBX120_2, 1920, 1208);
    }
    else
    {
      sync_inference(camera::id::top_left_front_120, h, &mat120_2, &vBBX120_2, 1920, 1208);
    }
  }
}

void callback_120_3(const sensor_msgs::Image::ConstPtr& msg)
{
  if (!isInferData_3)
  {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    mat120_3 = cv_ptr->image;
    std_msgs::Header h = msg->header;

    if (isCalibration)
    {
      if (input_resize)
      {
        cv::resize(mat120_3, mat120_3_resize, cv::Size(rawimg_w, rawimg_h));
        calibrationImage(mat120_3_resize, mat120_3_rect, cameraMatrix, distCoeffs);
      }
      else
      {
        calibrationImage(mat120_3, mat120_3_rect, cameraMatrix, distCoeffs);
      }
      sync_inference(camera::id::top_left_rear_120, h, &mat120_3_rect, &vBBX120_3, 1920, 1208);
    }
    else
    {
      sync_inference(camera::id::top_left_rear_120, h, &mat120_3, &vBBX120_3, 1920, 1208);
    }
  }
}

void callback_120_0_decode(sensor_msgs::CompressedImage compressImg)
{
  if (!isInferData_0)
  {
    cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(mat120_0);
    std_msgs::Header h = compressImg.header;

    if (isCalibration)
    {
      if (input_resize)
      {
        cv::resize(mat120_0, mat120_0_resize, cv::Size(rawimg_w, rawimg_h));
        calibrationImage(mat120_0_resize, mat120_0_rect, cameraMatrix, distCoeffs);
      }
      else
      {
        calibrationImage(mat120_0, mat120_0_rect, cameraMatrix, distCoeffs);
      }
      sync_inference(camera::id::top_right_front_120, h, &mat120_0_rect, &vBBX120_0, 1920, 1208);
    }
    else
    {
      sync_inference(camera::id::top_right_front_120, h, &mat120_0, &vBBX120_0, 1920, 1208);
    }
  }
}

void callback_120_1_decode(sensor_msgs::CompressedImage compressImg)
{
  if (!isInferData_1)
  {
    cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(mat120_1);
    std_msgs::Header h = compressImg.header;

    if (isCalibration)
    {
      if (input_resize)
      {
        cv::resize(mat120_1, mat120_1_resize, cv::Size(rawimg_w, rawimg_h));
        calibrationImage(mat120_1_resize, mat120_1_rect, cameraMatrix, distCoeffs);
      }
      else
      {
        calibrationImage(mat120_1, mat120_1_rect, cameraMatrix, distCoeffs);
      }
      sync_inference(camera::id::top_right_rear_120, h, &mat120_1_rect, &vBBX120_1, 1920, 1208);
    }
    else
    {
      sync_inference(camera::id::top_right_rear_120, h, &mat120_1, &vBBX120_1, 1920, 1208);
    }
  }
}

void callback_120_2_decode(sensor_msgs::CompressedImage compressImg)
{
  if (!isInferData_2)
  {
    cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(mat120_2);
    std_msgs::Header h = compressImg.header;

    if (isCalibration)
    {
      if (input_resize)
      {
        cv::resize(mat120_2, mat120_2_resize, cv::Size(rawimg_w, rawimg_h));
        calibrationImage(mat120_2_resize, mat120_2_rect, cameraMatrix, distCoeffs);
      }
      else
      {
        calibrationImage(mat120_2, mat120_2_rect, cameraMatrix, distCoeffs);
      }
      sync_inference(camera::id::top_left_front_120, h, &mat120_2_rect, &vBBX120_2, 1920, 1208);
    }
    else
    {
      sync_inference(camera::id::top_left_front_120, h, &mat120_2, &vBBX120_2, 1920, 1208);
    }
  }
}

void callback_120_3_decode(sensor_msgs::CompressedImage compressImg)
{
  if (!isInferData_3)
  {
    cv::imdecode(cv::Mat(compressImg.data), 1).copyTo(mat120_3);
    std_msgs::Header h = compressImg.header;

    if (isCalibration)
    {
      if (input_resize)
      {
        cv::resize(mat120_3, mat120_3_resize, cv::Size(rawimg_w, rawimg_h));
        calibrationImage(mat120_3_resize, mat120_3_rect, cameraMatrix, distCoeffs);
      }
      else
      {
        calibrationImage(mat120_3, mat120_3_rect, cameraMatrix, distCoeffs);
      }
      sync_inference(camera::id::top_left_rear_120, h, &mat120_3_rect, &vBBX120_3, 1920, 1208);
    }
    else
    {
      sync_inference(camera::id::top_left_rear_120, h, &mat120_3, &vBBX120_3, 1920, 1208);
    }
  }
}

void image_publisher(cv::Mat image, std_msgs::Header header, int cam_order)
{
  imgMsg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();

  if (cam_order == camera::id::top_right_front_120)
    pubImg_120_0.publish(imgMsg);
  else if (cam_order == camera::id::top_right_rear_120)
    pubImg_120_1.publish(imgMsg);
  else if (cam_order == camera::id::top_left_front_120)
    pubImg_120_2.publish(imgMsg);
  else if (cam_order == camera::id::top_left_rear_120)
    pubImg_120_3.publish(imgMsg);
}

/// roslaunch drivenet drivenet120.launch
int main(int argc, char** argv)
{
  ros::init(argc, argv, "drivenet_120_1_b1");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  isInferStop = false;
  isInferData = false;

  ros::param::get(ros::this_node::getName() + "/car_id", car_id);
  ros::param::get(ros::this_node::getName() + "/standard_fps", standard_FPS);
  ros::param::get(ros::this_node::getName() + "/display", display_flag);
  ros::param::get(ros::this_node::getName() + "/input_resize", input_resize);
  ros::param::get(ros::this_node::getName() + "/imgResult_publish", imgResult_publish);

  cam120_0_topicName = camera::topics[camera::id::top_right_front_120];
  cam120_1_topicName = camera::topics[camera::id::top_right_rear_120];
  cam120_2_topicName = camera::topics[camera::id::top_left_front_120];
  cam120_3_topicName = camera::topics[camera::id::top_left_rear_120];

  if (isCompressed)
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

  if (imgResult_publish)
  {
    pubImg_120_0 = it.advertise(cam120_0_topicName + std::string("/detect_image"), 1);
    pubImg_120_1 = it.advertise(cam120_1_topicName + std::string("/detect_image"), 1);
    pubImg_120_2 = it.advertise(cam120_2_topicName + std::string("/detect_image"), 1);
    pubImg_120_3 = it.advertise(cam120_3_topicName + std::string("/detect_image"), 1);
  }

  pub120_0 = nh.advertise<msgs::DetectedObjectArray>("/CamObjRightFront", 4);
  pub120_1 = nh.advertise<msgs::DetectedObjectArray>("/CamObjRightBack", 4);
  pub120_2 = nh.advertise<msgs::DetectedObjectArray>("/CamObjLeftFront", 4);
  pub120_3 = nh.advertise<msgs::DetectedObjectArray>("/CamObjLeftBack", 4);

  pthread_mutex_init(&mtxInfer, NULL);
  pthread_cond_init(&cndInfer, NULL);

  pthread_create(&thrdYolo, NULL, &run_yolo, NULL);
  if (standard_FPS == 1)
    pthread_create(&thrdInterp, NULL, &run_interp, NULL);
  if (display_flag == 1)
    pthread_create(&thrdDisplay, NULL, &run_display, NULL);

  std::string pkg_path = ros::package::getPath("drivenet");
  std::string cfg_file = "/b1_yolo_120_1.cfg";
  image_init();
  yoloApp.init_yolo(pkg_path, cfg_file);
  distEst.init(car_id);

  if (isCalibration)
  {
    cv::String calibMatrix_filepath = pkg_path + "/config/sf3324.yml";
    std::cout << "Calibration mode is open, calibMatrix filepath: " << calibMatrix_filepath << std::endl;
    loadCalibrationMatrix(calibMatrix_filepath, cameraMatrix, distCoeffs);
  }

  ros::MultiThreadedSpinner spinner(4);
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
    pub120_0.publish(doa120_0);
    pub120_1.publish(doa120_1);
    pub120_2.publish(doa120_2);
    pub120_3.publish(doa120_3);

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
  if (cam_order == camera::id::top_right_front_120)
  {
    // Right front 120 range:

    BoxPass_flag = false;
  }
  else if (cam_order == camera::id::top_right_rear_120)
  {
    // Right back 120 range:
    // x axis: 0 ~ 7 meters
    // y axis: -9 ~ 6 meters
    // cv::Point LeftLinePoint1(127, 272);
    // cv::Point LeftLinePoint2(-1422, 1207);
    // cv::Point RightLinePoint1(1904, 272);
    // cv::Point RightLinePoint2(3548, 1207);
    // BoxPass_flag = checkBoxInArea(RightLinePoint1, RightLinePoint2, LeftLinePoint1, LeftLinePoint2, box.x1, box.y2,
    // box.x2, box.y2);

    // if (box.y2 < 319) BoxPass_flag = false;
    BoxPass_flag = false;
  }
  else if (cam_order == camera::id::top_left_front_120)
  {
    // Left Front 120 range:
    BoxPass_flag = false;
  }
  else if (cam_order == camera::id::top_left_rear_120)
  {
    // Left back 120 range:
    BoxPass_flag = false;
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
  vBBX120_0.clear();
  vBBX120_1.clear();
  vBBX120_2.clear();
  vBBX120_3.clear();
  vbbx_output.clear();
  dist_cols.clear();
  dist_rows.clear();

  isInferData = false;
  isInferData_0 = false;
  isInferData_1 = false;
  isInferData_2 = false;
  isInferData_3 = false;
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
    if (!input_resize || isCalibration)
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
            float distMeter_p0x = 0, distMeter_p3x = 0, distMeter_p0y = 0, distMeter_p3y = 0;
            if (cam_order == camera::id::top_right_front_120 || cam_order == camera::id::top_right_rear_120)
            {
              distMeter_p0x = detObj.bPoint.p4.x;
              distMeter_p3x = detObj.bPoint.p0.x;
              distMeter_p0y = detObj.bPoint.p4.y;
              distMeter_p3y = detObj.bPoint.p0.y;
            }
            else if (cam_order == camera::id::top_left_front_120 || cam_order == camera::id::top_left_rear_120)
            {
              distMeter_p0x = detObj.bPoint.p3.x;
              distMeter_p3x = detObj.bPoint.p7.x;
              distMeter_p0y = detObj.bPoint.p3.y;
              distMeter_p3y = detObj.bPoint.p7.y;
            }

            float centerPoint[2]; 
            centerPoint[0] = (distMeter_p0x + distMeter_p3x)/2;
            centerPoint[1] = (distMeter_p0y + distMeter_p3y)/2;
            float distance = sqrt(pow(centerPoint[0], 2) + pow(centerPoint[1], 2));
            rounding(distance, 1);
            std::string distance_str = floatToString(distance);

            class_color = get_commonLabelColor(cls_color, detObj.classId);
            cv::putText(M_display, distance_str + " m",
                        cvPoint(x1 + 10, y1 - 10), 0, 1.5, class_color, 2);
          }
        }
      }
      doa.header = headers_tmp[ndx];
      doa.header.frame_id = "lidar";
      doa.objects = vDo;

      if (cam_order == camera::id::top_right_front_120)
      {
        if (standard_FPS == 1)
          doa120_0 = doa;
        else
          pub120_0.publish(doa);

        if (imgResult_publish || display_flag)
        {
          display_mutex.lock();
          mat120_0_display = M_display.clone();
          display_mutex.unlock();

          if (imgResult_publish)
          {
            image_publisher(mat120_0_display, headers_tmp[ndx], cam_order);
          }
        }
      }
      else if (cam_order == camera::id::top_right_rear_120)
      {
        if (standard_FPS == 1)
          doa120_1 = doa;
        else
          pub120_1.publish(doa);

        if (imgResult_publish || display_flag)
        {
          display_mutex.lock();
          mat120_1_display = M_display.clone();
          display_mutex.unlock();

          if (imgResult_publish)
          {
            image_publisher(mat120_1_display, headers_tmp[ndx], cam_order);
          }
        }
      }
      else if (cam_order == camera::id::top_left_front_120)
      {
        if (standard_FPS == 1)
          doa120_2 = doa;
        else
          pub120_2.publish(doa);

        if (imgResult_publish || display_flag)
        {
          display_mutex.lock();
          mat120_2_display = M_display.clone();
          display_mutex.unlock();

          if (imgResult_publish)
          {
            image_publisher(mat120_2_display, headers_tmp[ndx], cam_order);
          }
        }
      }
      else if (cam_order == camera::id::top_left_rear_120)
      {
        if (standard_FPS == 1)
          doa120_3 = doa;
        else
          pub120_3.publish(doa);

        if (imgResult_publish || display_flag)
        {
          display_mutex.lock();
          mat120_3_display = M_display.clone();
          display_mutex.unlock();

          if (imgResult_publish)
          {
            image_publisher(mat120_3_display, headers_tmp[ndx], cam_order);
          }
        }
      }
      vDo.clear();
    }

    std::cout << "Detect " << camera::topics[cam_ids_[0]] << ", " << camera::topics[cam_ids_[1]] << ", "
              << camera::topics[cam_ids_[2]] << " and " << camera::topics[cam_ids_[3]] << " image." << std::endl;

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
  // boundaryMarker(rawimg_w, boundaryMarker_0_1, boundaryMarker_0_2, boundaryMarker_0_3, boundaryMarker_0_4,
  // marker_h_0);
  // boundaryMarker(rawimg_w, boundaryMarker_1_1, boundaryMarker_1_2, boundaryMarker_1_3, boundaryMarker_1_4,
  // marker_h_1);
  // boundaryMarker(rawimg_w, boundaryMarker_2_1, boundaryMarker_2_2, boundaryMarker_2_3, boundaryMarker_2_4,
  // marker_h_2);
  // boundaryMarker(rawimg_w, boundaryMarker_3_1, boundaryMarker_3_2, boundaryMarker_3_3, boundaryMarker_3_4,
  // marker_h_3);

  ros::Rate r(10);
  while (ros::ok() && !isInferStop)
  {
    if (mat120_0_display.data || mat120_1_display.data || mat120_2_display.data || mat120_3_display.data)
    {
      if (mat120_0_display.cols * mat120_0_display.rows == rawimg_size &&
          mat120_1_display.cols * mat120_1_display.rows == rawimg_size &&
          mat120_2_display.cols * mat120_2_display.rows == rawimg_size &&
          mat120_3_display.cols * mat120_3_display.rows == rawimg_size)
      {
        try
        {
          display_mutex.lock();
          // cv::line(mat120_1_display, boundaryMarker_1_1, boundaryMarker_1_2, cv::Scalar(255, 255, 255), 1);
          // cv::line(mat120_1_display, boundaryMarker_1_3, boundaryMarker_1_4, cv::Scalar(255, 255, 255), 1);
          cv::imshow("RightFront-120", mat120_0_display);
          cv::imshow("RightBack-120", mat120_1_display);
          cv::imshow("LeftFront-120", mat120_2_display);
          cv::imshow("LeftBack-120", mat120_3_display);
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
