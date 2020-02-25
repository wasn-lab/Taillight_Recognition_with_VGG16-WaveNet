
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <signal.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "ros/ros.h"
#include "std_msgs/Header.h"
#include "std_msgs/String.h"
#include <msgs/PointXYZV.h>
#include <msgs/DetectedObjectArray.h>
#include <msgs/DetectedObject.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <string.h>
//#include <iostream>

// fps30
#include "ROSPublish.h"

#define CAMERA_DETECTION 0

#define LID_Front_Short 20
#define BB2BB_distance 4  // 3

#define max_det 64

#define fSize 9

static const int TOTAL_CB = 1;  // 4;//12;

/************************************************************************/
ros::Publisher pub;

int syncCount = 0;

void sync_callbackThreads();
pthread_mutex_t callback_mutex;
pthread_cond_t callback_cond;
/************************************************************************/
std_msgs::Header lidarHeader;
std_msgs::Header cam60_0_Header;
std_msgs::Header cam60_1_Header;
std_msgs::Header cam60_2_Header;
std_msgs::Header cam30_0_Header;
std_msgs::Header cam30_1_Header;
std_msgs::Header cam30_2_Header;
std_msgs::Header cam120_0_Header;
std_msgs::Header cam120_1_Header;
std_msgs::Header cam120_2_Header;
/************************************************************************/
msgs::DetectedObjectArray msgLidarObj;
msgs::DetectedObjectArray msgCam60_0_Obj;
msgs::DetectedObjectArray msgCam60_1_Obj;
msgs::DetectedObjectArray msgCam60_2_Obj;
msgs::DetectedObjectArray msgCam30_0_Obj;
msgs::DetectedObjectArray msgCam30_1_Obj;
msgs::DetectedObjectArray msgCam30_2_Obj;
msgs::DetectedObjectArray msgCam120_0_Obj;
msgs::DetectedObjectArray msgCam120_1_Obj;
msgs::DetectedObjectArray msgCam120_2_Obj;
/************************************************************************/

msgs::DetectedObjectArray msgFusionObj;
ros::Publisher fusion_pub;
std::thread publisher;

void fuseDetectedObjects();

/************************************************************************/
/******************put Lidar object to different view *******************/
/************************************************************************/

std::vector<msgs::DetectedObject> vDetectedObjectDF;
std::vector<msgs::DetectedObject> vDetectedObjectLID;
std::vector<msgs::DetectedObject> vDetectedObjectCAM_60_0;
std::vector<msgs::DetectedObject> vDetectedObjectCAM_60_1;
std::vector<msgs::DetectedObject> vDetectedObjectCAM_60_2;
std::vector<msgs::DetectedObject> vDetectedObjectTemp;
std::vector<msgs::DetectedObject> vDetectedObjectCAM_30_1;
std::vector<msgs::DetectedObject> vDetectedObjectCAM_120_1;

msgs::DetectedObjectArray msgLidar_60_0_Obj;
msgs::DetectedObjectArray msgLidar_60_1_Obj;
msgs::DetectedObjectArray msgLidar_60_2_Obj;
msgs::DetectedObjectArray msgLidar_30_0_Obj;
msgs::DetectedObjectArray msgLidar_30_1_Obj;
msgs::DetectedObjectArray msgLidar_30_2_Obj;
msgs::DetectedObjectArray msgLidar_120_0_Obj;
msgs::DetectedObjectArray msgLidar_120_1_Obj;
msgs::DetectedObjectArray msgLidar_120_2_Obj;
msgs::DetectedObjectArray msgLidar_others_Obj;
msgs::DetectedObjectArray msgLidar_rear_Obj;
msgs::DetectedObjectArray msgLidar_frontshort;

/**************************************************************************/
int** cam_det;
int** lid_det;

int total_det;
int** bb_det;
int total_det2;
int** bb_det2;
/**************************************************************************/

uint32_t seq = 0;

// fps30
typedef void (*PublishCallbackFunctionPtr)(void*, msgs::DetectedObjectArray&);
// The callback provided by the client via connectCallback().
PublishCallbackFunctionPtr mPublish_cb;
ROSPublish* rosPublisher;

/**************************************************************************/

void MySigintHandler(int sig)
{
  printf("****** MySigintHandler ******\n");
  ROS_INFO("shutting down!");
  std::thread::id this_id = std::this_thread::get_id();
  cout << this_id << endl;
  rosPublisher->stop();
  publisher.join();
  printf("after join()\n");
  ros::shutdown();
}

/************************************************************************/

void LidarDetectionCb(const msgs::DetectedObjectArray::ConstPtr& LidarObjArray)
{
  static std_msgs::Header pre_h;

  vDetectedObjectDF.clear();
  vDetectedObjectTemp.clear();
  vDetectedObjectLID.clear();

  msgLidarObj.header = LidarObjArray->header;

  if (pre_h.stamp.sec == LidarObjArray->header.stamp.sec && pre_h.stamp.nsec == LidarObjArray->header.stamp.nsec)
    return;
  else
    pre_h = LidarObjArray->header;

  msgLidarObj = *LidarObjArray;

  sync_callbackThreads();
}

void callback_camera_main(const msgs::DetectedObjectArray::ConstPtr& cam_obj_array,
                          msgs::DetectedObjectArray& msg_cam_obj)
{
  std::vector<msgs::DetectedObject> vDetectedObject;
  vDetectedObject.reserve(cam_obj_array->objects.size());

  for (size_t i = 0; i < cam_obj_array->objects.size(); i++)
  {
    if (cam_obj_array->objects[i].distance >= 0)
    {
      vDetectedObject.push_back(cam_obj_array->objects[i]);
    }
  }

  msg_cam_obj.header = cam_obj_array->header;
  msg_cam_obj.objects.assign(vDetectedObject.begin(), vDetectedObject.end());

  sync_callbackThreads();
}

void cam60_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam60_0_ObjArray)
{
  callback_camera_main(Cam60_0_ObjArray, msgCam60_0_Obj);
}

void cam60_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam60_1_ObjArray)
{
  callback_camera_main(Cam60_1_ObjArray, msgCam60_1_Obj);
}

void cam60_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam60_2_ObjArray)
{
  callback_camera_main(Cam60_2_ObjArray, msgCam60_2_Obj);
}

void cam30_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam30_0_ObjArray)
{
  callback_camera_main(Cam30_0_ObjArray, msgCam30_0_Obj);
}

void cam30_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam30_1_ObjArray)
{
  callback_camera_main(Cam30_1_ObjArray, msgCam30_1_Obj);
}

void cam30_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam30_2_ObjArray)
{
  callback_camera_main(Cam30_2_ObjArray, msgCam30_2_Obj);
}

void cam120_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam120_0_ObjArray)
{
  callback_camera_main(Cam120_0_ObjArray, msgCam120_0_Obj);
}

void cam120_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam120_1_ObjArray)
{
  callback_camera_main(Cam120_1_ObjArray, msgCam120_1_Obj);
}

void cam120_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam120_2_ObjArray)
{
  callback_camera_main(Cam120_2_ObjArray, msgCam120_2_Obj);
}

void fuseDetectedObjects()
{
  vDetectedObjectDF.clear();
  vDetectedObjectLID.clear();
  vDetectedObjectCAM_60_0.clear();
  vDetectedObjectCAM_60_1.clear();
  vDetectedObjectCAM_60_2.clear();
  vDetectedObjectCAM_30_1.clear();
  vDetectedObjectCAM_120_1.clear();

  /************************************************************************/

  for (auto& obj : msgLidarObj.objects)
  {
    obj.header = msgLidarObj.header;
    vDetectedObjectLID.push_back(obj);
  }
  printf("vDetectedObjectLID.size() = %zu \n", vDetectedObjectLID.size());

  /************************************************************************/

  for (auto& obj : msgCam60_0_Obj.objects)
  {
    obj.header = msgCam60_0_Obj.header;
    vDetectedObjectCAM_60_0.push_back(obj);
  }
  printf("vDetectedObjectCAM_60_0.size() = %zu \n", vDetectedObjectCAM_60_0.size());

  /************************************************************************/

  for (auto& obj : msgCam60_1_Obj.objects)
  {
    obj.header = msgCam60_1_Obj.header;
    vDetectedObjectCAM_60_1.push_back(obj);
  }
  printf("vDetectedObjectCAM_60_1.size() = %zu \n", vDetectedObjectCAM_60_1.size());

  /************************************************************************/

  for (auto& obj : msgCam60_2_Obj.objects)
  {
    obj.header = msgCam60_2_Obj.header;
    vDetectedObjectCAM_60_2.push_back(obj);
  }
  printf("vDetectedObjectCAM_60_2.size() = %zu \n", vDetectedObjectCAM_60_2.size());

  /************************************************************************/

  for (auto& obj : msgCam30_1_Obj.objects)
  {
    obj.header = msgCam30_1_Obj.header;
    vDetectedObjectCAM_30_1.push_back(obj);
  }
  printf("vDetectedObjectCAM_30_1.size() = %zu \n", vDetectedObjectCAM_30_1.size());

  /************************************************************************/

  for (auto& obj : msgCam120_1_Obj.objects)
  {
    obj.header = msgCam120_1_Obj.header;
    vDetectedObjectCAM_120_1.push_back(obj);
  }
  printf("vDetectedObjectCAM_120_1.size() = %zu \n", vDetectedObjectCAM_120_1.size());

  /************************************************************************/

  for (const auto& obj : vDetectedObjectLID)
  {
    vDetectedObjectDF.push_back(obj);
  }

  for (const auto& obj : vDetectedObjectCAM_60_0)
  {
    vDetectedObjectDF.push_back(obj);
  }

  for (const auto& obj : vDetectedObjectCAM_60_1)
  {
    vDetectedObjectDF.push_back(obj);
  }

  for (const auto& obj : vDetectedObjectCAM_60_2)
  {
    vDetectedObjectDF.push_back(obj);
  }

  for (const auto& obj : vDetectedObjectCAM_30_1)
  {
    vDetectedObjectDF.push_back(obj);
  }

  for (const auto& obj : vDetectedObjectCAM_120_1)
  {
    vDetectedObjectDF.push_back(obj);
  }

  msgFusionObj.objects = vDetectedObjectDF;
  msgFusionObj.header.stamp = msgLidarObj.header.stamp;
  msgFusionObj.header.frame_id = "lidar";
  msgFusionObj.header.seq = seq++;

  fusion_pub.publish(msgFusionObj);
}

void sync_callbackThreads()
{
  int tmp;
  cerr << __func__ << ":" << __LINE__ << endl;
  printf("****************************syncCount = %d****************************\n", syncCount);

  while (ros::ok())
  {
    pthread_mutex_lock(&callback_mutex);
    if (syncCount < TOTAL_CB - 1)
    {
      cerr << __func__ << ":" << __LINE__ << endl;
      syncCount++;

      pthread_cond_wait(&callback_cond, &callback_mutex);
    }
    else
    {
      cerr << __func__ << ":" << __LINE__ << endl;

      printf("****************************do_function****************************\n");

      fuseDetectedObjects();

      printf("****************************end do_function****************************\n");
      syncCount = 0;
      tmp = pthread_cond_broadcast(&callback_cond);
      printf("****************************pthread_cond_broadcast return %d****************************\n", tmp);
    }
    break;
  }
  pthread_mutex_unlock(&callback_mutex);
  cerr << __func__ << ":" << __LINE__ << endl;
}

void matrix_vector_multiply_3x3_3d(double m[9], double v[3], double result[3])
{
  result[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
  result[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
  result[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

void vector_add_3d(double v1[3], double v2[3], double result[3])
{
  result[0] = v1[0] + v2[0];
  result[1] = v1[1] + v2[1];
  result[2] = v1[2] + v2[2];
}

int main(int argc, char** argv)
{
  cam_det = new int*[5];
  for (int i = 0; i < 5; i++)
  {
    cam_det[i] = (int*)malloc(sizeof(int) * max_det);
    memset(cam_det[i], 0, sizeof(int) * max_det);
  }

  lid_det = new int*[5];
  for (int i = 0; i < 5; i++)
  {
    lid_det[i] = (int*)malloc(sizeof(int) * max_det);
    memset(lid_det[i], 0, sizeof(int) * max_det);
  }

  bb_det = new int*[6];
  for (int i = 0; i < 6; i++)
  {
    bb_det[i] = (int*)malloc(sizeof(int) * (3 * max_det));
    memset(bb_det[i], 0, sizeof(int) * (3 * max_det));
  }

  // Variables for Fused Detection

  bb_det2 = new int*[6];
  for (int i = 0; i < 6; i++)
  {
    bb_det2[i] = (int*)malloc(sizeof(int) * (3 * max_det));
    memset(bb_det2[i], 0, sizeof(int) * (3 * max_det));
  }

  /**************************************************************************/

  ros::init(argc, argv, "sensor_fusion");
  ros::NodeHandle nh;

  // Lidar object detection input
  ros::Subscriber lidar_det_sub = nh.subscribe("/LidarDetection", 2, LidarDetectionCb);

// Camera object detection input
#if CAMERA_DETECTION == 1
  ros::Subscriber cam_det_sub = nh.subscribe("/CameraDetection", 1, cam60_1_DetectionCb);
#else
  ros::Subscriber cam_F_right_sub = nh.subscribe("/CamObjFrontRight", 1, cam60_0_DetectionCb);
  ros::Subscriber cam_F_center_sub = nh.subscribe("/CamObjFrontCenter", 1, cam60_1_DetectionCb);
  ros::Subscriber cam_F_left_sub = nh.subscribe("/CamObjFrontLeft", 1, cam60_2_DetectionCb);
#endif

  fusion_pub = nh.advertise<msgs::DetectedObjectArray>("SensorFusion", 2);

  syncCount = 0;
  pthread_mutex_init(&callback_mutex, NULL);
  pthread_cond_init(&callback_cond, NULL);

  signal(SIGINT, MySigintHandler);

  ros::MultiThreadedSpinner spinner(TOTAL_CB);
  spinner.spin();

  /*******************************************************/

  for (int i = 0; i < 5; i++)
  {
    free(cam_det[i]);
    free(lid_det[i]);
  }

  for (int i = 0; i < 6; i++)
  {
    free(bb_det[i]);
    free(bb_det2[i]);
  }

  printf("***********free memory 3**************\n");
}
