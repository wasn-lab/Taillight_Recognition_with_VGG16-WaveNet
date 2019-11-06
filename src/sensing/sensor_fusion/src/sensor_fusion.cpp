
#include <math.h>
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
#include <msgs/Rad.h>
#include <msgs/PointXYZV.h>
#include <msgs/DetectedObjectArray.h>
#include <msgs/DetectedObject.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
//#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>

//#include "radar1T2R.h"
#include <string.h>
//#include <iostream>

// fps30
#include "ROSPublish.h"

#define LID_Front_Short 20
#define BB2BB_distance 4  // 3

#define max_det 64

#define fSize 9
#define PI 3.14159265358979

/************************************************************************/

#define radar_coordinate_offset_X 1.41
#define radar_coordinate_offset_Y 0
#define radar_coordinate_offset_Z 0.875

#define lidar_coordinate_offset_X 0.4
#define lidar_coordinate_offset_Y 0
#define lidar_coordinate_offset_Z 3.42

#define cam_coordinate_offset_X 0.4
#define cam_coordinate_offset_Y 0
#define cam_coordinate_offset_Z 3.42
/************************************************************************/

//#define EnableFusion
#define EnableLIDAR
//#define EnableRADAR

static const int TOTAL_CB = 1;  // 4;//12;

/*
#define EnableRADAR
#define EnableLIDAR
#define EnableImage
#define EnableCAM30_0
#define EnableCAM30_1
#define EnableCAM30_2
#define EnableCAM60_0
#define EnableCAM60_1
#define EnableCAM60_2
#define EnableCAM120_0
#define EnableCAM120_1
#define EnableCAM120_2
*/

/************************************************************************/
double K_Matrix[9] = { 1725, 0, 900, 0, 1500, 920, 0, 0, 1 };
double t_Vector[3] = { 0, 1.7, 0 };
double R_Matrix[9] = { 0.999999, 0.000495691, 0.00167342, 0.0017453, -0.284015, -0.958818, 0, 0.95882, -0.284015 };

double temp[3];
double radar_xyz[3];
double camera_xyz[3];
double image_point[3];
/************************************************************************/
void transform_coordinate(msgs::PointXYZ& p, const float x, const float y, const float z);
void transform_coordinate_main(msgs::ConvexPoint& cp, const float x, const float y, const float z);

uint32_t dbgPCView;
pthread_mutex_t mut_dbgPCView;
pthread_cond_t cnd_dbgPCView;
pthread_t thrd_dbgPCView;
void* dbg_drawPointCloud(void* arg);

ros::Publisher pub;

int wait_rad_cam = 0;
int syncCount = 0;
;
void sync_callbackThreads();
pthread_mutex_t callback_mutex;
pthread_cond_t callback_cond;

cv::Mat InImage;
cv::Mat InImage2;
bool isLaneCloudUpdate;
long long int lid_timestamp[2];
long long int rad_timestamp[2];

/************************************************************************/
int drawing_uv[max_det][4];
int drawing_num = 0;

int drawing_uv_cam[max_det][4];
int drawing_num_cam = 0;

int drawing_uv_lidar[max_det][4];
int drawing_num_lidar = 0;

int radar_uv[max_det][4];
int radar_num = 0;

int Lidar_uv[max_det][4];
int Lidar_num = 0;

int Cam60_0_uv[max_det][4];
int Cam60_0_num = 0;
int Cam60_1_uv[max_det][4];
int Cam60_1_num = 0;
int Cam60_2_uv[max_det][4];
int Cam60_2_num = 0;

int Cam30_0_uv[max_det][4];
int Cam30_0_num = 0;
int Cam30_1_uv[max_det][4];
int Cam30_1_num = 0;
int Cam30_2_uv[max_det][4];
int Cam30_2_num = 0;

int Cam120_0_uv[max_det][4];
int Cam120_0_num = 0;
int Cam120_1_uv[max_det][4];
int Cam120_1_num = 0;
int Cam120_2_uv[max_det][4];
int Cam120_2_num = 0;
/************************************************************************/
int Cam60_1_num_cb = 0;
int radar_num_cb = 0;
int Lidar_num_cb = 0;
int Cam30_1_num_cb = 0;
int Cam120_1_num_cb = 0;
/************************************************************************/
msgs::DetectedObjectArray msgRadObj;

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
ros::Publisher fusMsg_pub;
std::thread publisher;

void RadarDetectionCb(const msgs::DetectedObjectArray::ConstPtr& RadObjArray);
void LidarDetectionCb(const msgs::DetectedObjectArray::ConstPtr& RadObjArray);
void cam60_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& RadObjArray);
void cam60_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& RadObjArray);
void cam60_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& RadObjArray);
void cam30_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& RadObjArray);
void cam30_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& RadObjArray);
void cam30_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& RadObjArray);
void cam120_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& RadObjArray);
void cam120_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& RadObjArray);
void cam120_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& RadObjArray);

void decisionFusion();
void decision3DFusion();

void Cam30_0_view_fusion(void);
void Cam30_1_view_fusion(void);
void Cam30_2_view_fusion(void);

void Cam60_0_view_fusion(void);
void Cam60_1_view_fusion(void);
void Cam60_2_view_fusion(void);

void Cam120_0_view_fusion(void);
void Cam120_1_view_fusion(void);
void Cam120_2_view_fusion(void);

/************************************************************************/
/******************put Lidar object to different view *******************/
/************************************************************************/

std::vector<msgs::DetectedObject> vDetectedObjectDF;
std::vector<msgs::DetectedObject> vDetectedObjectRAD;
std::vector<msgs::DetectedObject> vDetectedObjectLID;
std::vector<msgs::DetectedObject> vDetectedObjectCAM_60_1;
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

void overlap_analysis(int** bb_det, int total_det);
void overlap_fusion(int** cam, int ncam, int** rad, int nrad, int** det, int* total_det);

/**************************************************************************/
int** cam_det;
int** lid_det;
int** radar_det;

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
int fusion_3D_to_2D(double x_3d, double y_3d, double z_3d, int* u_2d, int* v_2d);

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

int main(int argc, char** argv)
{
  cam_det = new int*[5];
  for (int j = 0; j < 5; j++)
  {
    cam_det[j] = (int*)malloc(sizeof(int) * max_det);
    memset(cam_det[j], 0, sizeof(int) * max_det);
  }

  lid_det = new int*[5];
  for (int j = 0; j < 5; j++)
  {
    lid_det[j] = (int*)malloc(sizeof(int) * max_det);
    memset(lid_det[j], 0, sizeof(int) * max_det);
  }

  radar_det = new int*[5];
  for (int j = 0; j < 5; j++)
  {
    radar_det[j] = (int*)malloc(sizeof(int) * max_det);
    memset(radar_det[j], 0, sizeof(int) * max_det);
  }

  bb_det = new int*[6];
  for (int j = 0; j < 6; j++)
  {
    bb_det[j] = (int*)malloc(sizeof(int) * (3 * max_det));
    memset(bb_det[j], 0, sizeof(int) * (3 * max_det));
  }

  // Variables for Fused Detection

  bb_det2 = new int*[6];
  for (int j = 0; j < 6; j++)
  {
    bb_det2[j] = (int*)malloc(sizeof(int) * (3 * max_det));
    memset(bb_det2[j], 0, sizeof(int) * (3 * max_det));
  }

  /**************************************************************************/

  ros::init(argc, argv, "sensor_fusion");
  ros::NodeHandle nhFus;
  // cv::namedWindow("BeforeFusion",CV_WINDOW_NORMAL);
  // cv::namedWindow("AfterFusion",CV_WINDOW_NORMAL);

  // Radar object detection input
  ros::Subscriber RadarDetectionSub = nhFus.subscribe("/RadarDetection", 2, RadarDetectionCb);

  // Lidar object detection input
  ros::Subscriber LidarDetectionSub = nhFus.subscribe("/LidarDetection", 2, LidarDetectionCb);

  // Camera object detection input
  ros::Subscriber cam60_1_DetectionSub = nhFus.subscribe("/DetectedObjectArray/cam60", 2, cam60_1_DetectionCb);
  ros::Subscriber cam30_1_DetectionSub = nhFus.subscribe("/DetectedObjectArray/cam30", 2, cam30_1_DetectionCb);
  ros::Subscriber cam120_1_DetectionSub = nhFus.subscribe("/DetectedObjectArray/cam120", 2, cam120_1_DetectionCb);

  fusMsg_pub = nhFus.advertise<msgs::DetectedObjectArray>("SensorFusion", 2);

  syncCount = 0;
  pthread_mutex_init(&callback_mutex, NULL);
  pthread_cond_init(&callback_cond, NULL);

  dbgPCView = 0;
  // pthread_mutex_init(&mut_dbgPCView,NULL);
  // pthread_cond_init(&cnd_dbgPCView,NULL);
  // pthread_create(&thrd_dbgPCView, NULL, &dbg_drawPointCloud, NULL);

  // ros::spin();

  // fps30
  // rosPublisher = new ROSPublish();
  // publisher = std::thread(&ROSPublish::tickFuntion, rosPublisher);
  // mPublish_cb = ROSPublish::staticPublishCallbackFunction;

  signal(SIGINT, MySigintHandler);

  ros::MultiThreadedSpinner spinner(TOTAL_CB);
  spinner.spin();  // spin() will not return until the node has been shutdown

  /*******************************************************/

  for (int j = 0; j < 5; j++)
    free(cam_det[j]);

  for (int j = 0; j < 5; j++)
    free(lid_det[j]);

  for (int j = 0; j < 5; j++)
    free(radar_det[j]);

  for (int j = 0; j < 6; j++)
    free(bb_det[j]);

  for (int j = 0; j < 6; j++)
    free(bb_det2[j]);

  printf("***********free memory 3**************\n");
  /******************************************************/
  return 0;
}

/************************************************************************/
/*****************************ITRI-DriveNet******************************/
/************************************************************************/
std_msgs::Header cam60_0_Header;
void cam60_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam60_0_ObjArray)
{
  // std::cerr << __func__ << ":" << __LINE__ << std::endl;

  int i, j, tmp, a, b, c, d;

  std::vector<msgs::DetectedObject> vDetectedObject = Cam60_0_ObjArray->objects;
  msgCam60_0_Obj.header = Cam60_0_ObjArray->header;
  // printf("Cam60_0_ObjArray->objects.size() = %ld\n",Cam60_0_ObjArray->objects.size());

  Cam60_0_num = Cam60_0_ObjArray->objects.size();

  for (i = 0; i < Cam60_0_ObjArray->objects.size(); i++)
  {
    Cam60_0_uv[i][0] = vDetectedObject[i].camInfo.u;
    Cam60_0_uv[i][1] = vDetectedObject[i].camInfo.v;
    Cam60_0_uv[i][2] = vDetectedObject[i].camInfo.width;
    Cam60_0_uv[i][3] = vDetectedObject[i].camInfo.height;
  }

  msgCam60_0_Obj = *Cam60_0_ObjArray;  // for fusion

#ifdef EnableCAM60_0
  sync_callbackThreads();
#endif
}

/************************************************************************/
std_msgs::Header cam60_1_Header;
void cam60_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam60_1_ObjArray)
{
  // std::cerr << __func__ << ":" << __LINE__ << std::endl;

  int i, j, tmp, a, b, c, d;

  vDetectedObjectCAM_60_1.clear();
  std::vector<msgs::DetectedObject> vDetectedObject = Cam60_1_ObjArray->objects;
  msgCam60_1_Obj.header = Cam60_1_ObjArray->header;
  // printf("Cam60_1_ObjArray->objects.size() = %ld\n",Cam60_1_ObjArray->objects.size());

  if (Cam60_1_ObjArray->objects.size() > max_det)
    Cam60_1_num_cb = max_det;
  else
    Cam60_1_num_cb = Cam60_1_ObjArray->objects.size();

  for (i = 0; i < Cam60_1_num_cb; i++)
  {
    Cam60_1_uv[i][0] = vDetectedObject[i].camInfo.u;
    Cam60_1_uv[i][1] = vDetectedObject[i].camInfo.v;
    Cam60_1_uv[i][2] = vDetectedObject[i].camInfo.width;
    Cam60_1_uv[i][3] = vDetectedObject[i].camInfo.height;
  }

  msgCam60_1_Obj = *Cam60_1_ObjArray;  // for fusion

#ifdef EnableCAM60_1
  sync_callbackThreads();
#endif
}
/************************************************************************/
std_msgs::Header cam60_2_Header;
void cam60_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam60_2_ObjArray)
{
  // std::cerr << __func__ << ":" << __LINE__ << std::endl;

  int i, j, tmp, a, b, c, d;

  std::vector<msgs::DetectedObject> vDetectedObject = Cam60_2_ObjArray->objects;
  msgCam60_2_Obj.header = Cam60_2_ObjArray->header;
  // printf("Cam60_2_ObjArray->objects.size() = %ld\n",Cam60_2_ObjArray->objects.size());

  Cam60_2_num = Cam60_2_ObjArray->objects.size();

  for (i = 0; i < Cam60_2_ObjArray->objects.size(); i++)
  {
    Cam60_2_uv[i][0] = vDetectedObject[i].camInfo.u;
    Cam60_2_uv[i][1] = vDetectedObject[i].camInfo.v;
    Cam60_2_uv[i][2] = vDetectedObject[i].camInfo.width;
    Cam60_2_uv[i][3] = vDetectedObject[i].camInfo.height;
  }

  msgCam60_2_Obj = *Cam60_2_ObjArray;  // for fusion

#ifdef EnableCAM60_2
  sync_callbackThreads();
#endif
}
/************************************************************************/
/************************************************************************/

std_msgs::Header cam30_0_Header;
void cam30_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam30_0_ObjArray)
{
  // std::cerr << __func__ << ":" << __LINE__ << std::endl;

  int i, j, tmp, a, b, c, d;

  std::vector<msgs::DetectedObject> vDetectedObject = Cam30_0_ObjArray->objects;
  msgCam30_0_Obj.header = Cam30_0_ObjArray->header;
  // printf("Cam30_0_ObjArray->objects.size() = %ld\n",Cam30_0_ObjArray->objects.size());

  Cam30_0_num = Cam30_0_ObjArray->objects.size();

  for (i = 0; i < Cam30_0_ObjArray->objects.size(); i++)
  {
    Cam30_0_uv[i][0] = vDetectedObject[i].camInfo.u;
    Cam30_0_uv[i][1] = vDetectedObject[i].camInfo.v;
    Cam30_0_uv[i][2] = vDetectedObject[i].camInfo.width;
    Cam30_0_uv[i][3] = vDetectedObject[i].camInfo.height;
  }

  msgCam30_0_Obj = *Cam30_0_ObjArray;  // for fusion

#ifdef EnableCAM30_0
  sync_callbackThreads();
#endif
}

/************************************************************************/
std_msgs::Header cam30_1_Header;
void cam30_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam30_1_ObjArray)
{
  // std::cerr << __func__ << ":" << __LINE__ << std::endl;

  int i, j, tmp, a, b, c, d;

  vDetectedObjectCAM_30_1.clear();
  std::vector<msgs::DetectedObject> vDetectedObject = Cam30_1_ObjArray->objects;
  msgCam30_1_Obj.header = Cam30_1_ObjArray->header;
  // printf("Cam30_1_ObjArray->objects.size() = %ld\n",Cam30_1_ObjArray->objects.size());

  if (Cam30_1_ObjArray->objects.size() > max_det)
    Cam30_1_num_cb = max_det;
  else
    Cam30_1_num_cb = Cam30_1_ObjArray->objects.size();

  msgCam30_1_Obj = *Cam30_1_ObjArray;  // for fusion

#ifdef EnableCAM30_1
  sync_callbackThreads();
#endif
}
/************************************************************************/
std_msgs::Header cam30_2_Header;
void cam30_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam30_2_ObjArray)
{
  // std::cerr << __func__ << ":" << __LINE__ << std::endl;

  int i, j, tmp, a, b, c, d;

  std::vector<msgs::DetectedObject> vDetectedObject = Cam30_2_ObjArray->objects;
  msgCam30_2_Obj.header = Cam30_2_ObjArray->header;
  // printf("Cam30_2_ObjArray->objects.size() = %ld\n",Cam30_2_ObjArray->objects.size());

  Cam30_2_num = Cam30_2_ObjArray->objects.size();

  for (i = 0; i < Cam30_2_ObjArray->objects.size(); i++)
  {
    Cam30_2_uv[i][0] = vDetectedObject[i].camInfo.u;
    Cam30_2_uv[i][1] = vDetectedObject[i].camInfo.v;
    Cam30_2_uv[i][2] = vDetectedObject[i].camInfo.width;
    Cam30_2_uv[i][3] = vDetectedObject[i].camInfo.height;
  }

  msgCam30_2_Obj = *Cam30_2_ObjArray;  // for fusion

#ifdef EnableCAM30_2
  sync_callbackThreads();
#endif
}

/************************************************************************/
/************************************************************************/

std_msgs::Header cam120_0_Header;
void cam120_0_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam120_0_ObjArray)
{
  // std::cerr << __func__ << ":" << __LINE__ << std::endl;

  int i, j, tmp, a, b, c, d;

  std::vector<msgs::DetectedObject> vDetectedObject = Cam120_0_ObjArray->objects;
  msgCam120_0_Obj.header = Cam120_0_ObjArray->header;
  // printf("Cam120_0_ObjArray->objects.size() = %ld\n",Cam120_0_ObjArray->objects.size());

  Cam120_0_num = Cam120_0_ObjArray->objects.size();

  for (i = 0; i < Cam120_0_ObjArray->objects.size(); i++)
  {
    Cam120_0_uv[i][0] = vDetectedObject[i].camInfo.u;
    Cam120_0_uv[i][1] = vDetectedObject[i].camInfo.v;
    Cam120_0_uv[i][2] = vDetectedObject[i].camInfo.width;
    Cam120_0_uv[i][3] = vDetectedObject[i].camInfo.height;
  }

  msgCam120_0_Obj = *Cam120_0_ObjArray;  // for fusion

#ifdef EnableCAM120_0
  sync_callbackThreads();
#endif
}

/************************************************************************/
std_msgs::Header cam120_1_Header;
void cam120_1_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam120_1_ObjArray)
{
  // std::cerr << __func__ << ":" << __LINE__ << std::endl;

  int i, j, tmp, a, b, c, d;

  vDetectedObjectCAM_120_1.clear();
  std::vector<msgs::DetectedObject> vDetectedObject = Cam120_1_ObjArray->objects;
  msgCam120_1_Obj.header = Cam120_1_ObjArray->header;
  // printf("Cam120_1_ObjArray->objects.size() = %ld\n",Cam120_1_ObjArray->objects.size());

  Cam120_1_num_cb = Cam120_1_ObjArray->objects.size();

  if (Cam120_1_ObjArray->objects.size() > max_det)
    Cam120_1_num_cb = max_det;
  else
    Cam120_1_num_cb = Cam120_1_ObjArray->objects.size();

  msgCam120_1_Obj = *Cam120_1_ObjArray;  // for fusion

#ifdef EnableCAM120_1
  sync_callbackThreads();
#endif
}
/************************************************************************/
std_msgs::Header cam120_2_Header;
void cam120_2_DetectionCb(const msgs::DetectedObjectArray::ConstPtr& Cam120_2_ObjArray)
{
  // std::cerr << __func__ << ":" << __LINE__ << std::endl;

  int i, j, tmp, a, b, c, d;

  std::vector<msgs::DetectedObject> vDetectedObject = Cam120_2_ObjArray->objects;
  msgCam120_2_Obj.header = Cam120_2_ObjArray->header;
  // printf("Cam120_2_ObjArray->objects.size() = %ld\n",Cam120_2_ObjArray->objects.size());

  Cam120_2_num = Cam120_2_ObjArray->objects.size();

  for (i = 0; i < Cam120_2_ObjArray->objects.size(); i++)
  {
    Cam120_2_uv[i][0] = vDetectedObject[i].camInfo.u;
    Cam120_2_uv[i][1] = vDetectedObject[i].camInfo.v;
    Cam120_2_uv[i][2] = vDetectedObject[i].camInfo.width;
    Cam120_2_uv[i][3] = vDetectedObject[i].camInfo.height;
  }

  msgCam120_2_Obj = *Cam120_2_ObjArray;  // for fusion

#ifdef EnableCAM120_2
  sync_callbackThreads();
#endif
}

/************************************************************************/
/*****************************ITRI-DriveNet******************************/
/************************************************************************/

std_msgs::Header lidarHeader;
void LidarDetectionCb(const msgs::DetectedObjectArray::ConstPtr& LidarObjArray)
{
  // std::cerr << __func__ << ":" << __LINE__ << std::endl;
  static std_msgs::Header pre_h;

  vDetectedObjectDF.clear();
  vDetectedObjectTemp.clear();
  vDetectedObjectLID.clear();

  int i, j, tmp, a, b, c, d;

  std::vector<msgs::DetectedObject> vDetectedObject = LidarObjArray->objects;
  msgLidarObj.header = LidarObjArray->header;

  if (pre_h.stamp.sec == LidarObjArray->header.stamp.sec && pre_h.stamp.nsec == LidarObjArray->header.stamp.nsec)
    return;
  else
    pre_h = LidarObjArray->header;

  if (LidarObjArray->objects.size() > max_det)
    Lidar_num_cb = max_det;
  else
    Lidar_num_cb = LidarObjArray->objects.size();

  msgLidarObj = *LidarObjArray;  // for fusion

#ifdef EnableLIDAR
  sync_callbackThreads();
#endif
}

std_msgs::Header radHeader;
void RadarDetectionCb(const msgs::DetectedObjectArray::ConstPtr& RadObjArray)
{
  // std::cerr << __func__ << ":" << __LINE__ << std::endl;

  int i, j, tmp, a, b, c, d;
  vDetectedObjectRAD.clear();
  std::vector<msgs::DetectedObject> vDetectedObject = RadObjArray->objects;

  msgRadObj.header = RadObjArray->header;

  if (RadObjArray->objects.size() > max_det)
    radar_num_cb = max_det;
  else
    radar_num_cb = RadObjArray->objects.size();

  for (i = 0; i < radar_num_cb; i++)
  {
    radar_uv[i][0] = vDetectedObject[i].radarInfo.imgPoint60.x;
    radar_uv[i][1] = vDetectedObject[i].radarInfo.imgPoint60.y;
    radar_uv[i][2] = 50;
    radar_uv[i][3] = 50;
  }

  msgRadObj = *RadObjArray;  // for fusion

#ifdef EnableRADAR
  sync_callbackThreads();
#endif
}

void decisionFusion()
{
  int tmp, a, b, c, d;
  int i, j;
  float p0x, p0y, p3x, p3y, cx, cy;
  float theta;

  std::vector<msgs::DetectedObject> vLidar_30_0_Object;
  std::vector<msgs::DetectedObject> vLidar_30_1_Object;
  std::vector<msgs::DetectedObject> vLidar_30_2_Object;

  std::vector<msgs::DetectedObject> vLidar_60_0_Object;
  std::vector<msgs::DetectedObject> vLidar_60_1_Object;
  std::vector<msgs::DetectedObject> vLidar_60_2_Object;

  std::vector<msgs::DetectedObject> vLidar_120_0_Object;
  std::vector<msgs::DetectedObject> vLidar_120_1_Object;
  std::vector<msgs::DetectedObject> vLidar_120_2_Object;
  std::vector<msgs::DetectedObject> vLidar_others_Object;
  std::vector<msgs::DetectedObject> vLidar_rear_Object;
  std::vector<msgs::DetectedObject> vLidar_frontshort_Object;

  vDetectedObjectDF.clear();
  vDetectedObjectRAD.clear();
  vDetectedObjectLID.clear();
  vDetectedObjectCAM_60_1.clear();
  vDetectedObjectCAM_30_1.clear();
  vDetectedObjectCAM_120_1.clear();

  vLidar_frontshort_Object.clear();
  vLidar_60_0_Object.clear();
  vLidar_60_2_Object.clear();
  vLidar_60_1_Object.clear();

  /************************************************************************/
  printf("Lidar_num_cb = %d \n", Lidar_num_cb);

  for (int j = 0; j < Lidar_num_cb; j++)
  {
    msgLidarObj.objects[j].header = msgLidarObj.header;  // add timestamp for lidar object

    msgLidarObj.objects[j].bPoint.p0.x = msgLidarObj.objects[j].bPoint.p0.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p0.y = msgLidarObj.objects[j].bPoint.p0.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p0.z = msgLidarObj.objects[j].bPoint.p0.z + lidar_coordinate_offset_Z;

    msgLidarObj.objects[j].bPoint.p1.x = msgLidarObj.objects[j].bPoint.p1.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p1.y = msgLidarObj.objects[j].bPoint.p1.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p1.z = msgLidarObj.objects[j].bPoint.p1.z + lidar_coordinate_offset_Z;

    msgLidarObj.objects[j].bPoint.p2.x = msgLidarObj.objects[j].bPoint.p2.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p2.y = msgLidarObj.objects[j].bPoint.p2.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p2.z = msgLidarObj.objects[j].bPoint.p2.z + lidar_coordinate_offset_Z;

    msgLidarObj.objects[j].bPoint.p3.x = msgLidarObj.objects[j].bPoint.p3.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p3.y = msgLidarObj.objects[j].bPoint.p3.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p3.z = msgLidarObj.objects[j].bPoint.p3.z + lidar_coordinate_offset_Z;

    msgLidarObj.objects[j].bPoint.p4.x = msgLidarObj.objects[j].bPoint.p4.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p4.y = msgLidarObj.objects[j].bPoint.p4.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p4.z = msgLidarObj.objects[j].bPoint.p4.z + lidar_coordinate_offset_Z;

    msgLidarObj.objects[j].bPoint.p5.x = msgLidarObj.objects[j].bPoint.p5.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p5.y = msgLidarObj.objects[j].bPoint.p5.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p5.z = msgLidarObj.objects[j].bPoint.p5.z + lidar_coordinate_offset_Z;

    msgLidarObj.objects[j].bPoint.p6.x = msgLidarObj.objects[j].bPoint.p6.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p6.y = msgLidarObj.objects[j].bPoint.p6.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p6.z = msgLidarObj.objects[j].bPoint.p6.z + lidar_coordinate_offset_Z;

    msgLidarObj.objects[j].bPoint.p7.x = msgLidarObj.objects[j].bPoint.p7.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p7.y = msgLidarObj.objects[j].bPoint.p7.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p7.z = msgLidarObj.objects[j].bPoint.p7.z + lidar_coordinate_offset_Z;
    transform_coordinate_main(msgLidarObj.objects[j].cPoint, lidar_coordinate_offset_X, lidar_coordinate_offset_Y,
                              lidar_coordinate_offset_Z);

    // ======>
    vDetectedObjectLID.push_back(msgLidarObj.objects[j]);
  }

  Lidar_num = vDetectedObjectLID.size();
  printf("vDetectedObjectRAD.size() = %ld \n", vDetectedObjectLID.size());

  /************************************************************************/

  /************************************************************************/
  printf("radar_num_cb = %d \n", radar_num_cb);

  for (int j = 0; j < msgRadObj.objects.size(); j++)
  {
    msgRadObj.objects[j].header = msgRadObj.header;  // add timestamp for radar object

    msgRadObj.objects[j].bPoint.p0.x = msgRadObj.objects[j].bPoint.p0.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p0.y = msgRadObj.objects[j].bPoint.p0.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p0.z = msgRadObj.objects[j].bPoint.p0.z + radar_coordinate_offset_Z;

    msgRadObj.objects[j].bPoint.p1.x = msgRadObj.objects[j].bPoint.p1.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p1.y = msgRadObj.objects[j].bPoint.p1.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p1.z = msgRadObj.objects[j].bPoint.p1.z + radar_coordinate_offset_Z;

    msgRadObj.objects[j].bPoint.p2.x = msgRadObj.objects[j].bPoint.p2.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p2.y = msgRadObj.objects[j].bPoint.p2.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p2.z = msgRadObj.objects[j].bPoint.p2.z + radar_coordinate_offset_Z;

    msgRadObj.objects[j].bPoint.p3.x = msgRadObj.objects[j].bPoint.p3.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p3.y = msgRadObj.objects[j].bPoint.p3.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p3.z = msgRadObj.objects[j].bPoint.p3.z + radar_coordinate_offset_Z;

    msgRadObj.objects[j].bPoint.p4.x = msgRadObj.objects[j].bPoint.p4.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p4.y = msgRadObj.objects[j].bPoint.p4.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p4.z = msgRadObj.objects[j].bPoint.p4.z + radar_coordinate_offset_Z;

    msgRadObj.objects[j].bPoint.p5.x = msgRadObj.objects[j].bPoint.p5.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p5.y = msgRadObj.objects[j].bPoint.p5.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p5.z = msgRadObj.objects[j].bPoint.p5.z + radar_coordinate_offset_Z;

    msgRadObj.objects[j].bPoint.p6.x = msgRadObj.objects[j].bPoint.p6.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p6.y = msgRadObj.objects[j].bPoint.p6.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p6.z = msgRadObj.objects[j].bPoint.p6.z + radar_coordinate_offset_Z;

    msgRadObj.objects[j].bPoint.p7.x = msgRadObj.objects[j].bPoint.p7.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p7.y = msgRadObj.objects[j].bPoint.p7.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p7.z = msgRadObj.objects[j].bPoint.p7.z + radar_coordinate_offset_Z;
    /************************************************************************/

    transform_coordinate_main(msgRadObj.objects[j].cPoint, radar_coordinate_offset_X, radar_coordinate_offset_Y,
                              radar_coordinate_offset_Z);

    /************************************************************************/
    // ======>
    vDetectedObjectRAD.push_back(msgRadObj.objects[j]);
  }

  radar_num = vDetectedObjectRAD.size();
  printf("vDetectedObjectRAD.size() = %ld \n", vDetectedObjectRAD.size());
  /************************************************************************/

  /************************************************************************/
  printf("Cam60_1_num_cb = %d \n", Cam60_1_num_cb);

  for (int j = 0; j < Cam60_1_num_cb; j++)
  {
    msgCam60_1_Obj.objects[j].header = msgCam60_1_Obj.header;  // add timestamp for cam60 object

    msgCam60_1_Obj.objects[j].bPoint.p0.x = msgCam60_1_Obj.objects[j].bPoint.p0.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p0.y = msgCam60_1_Obj.objects[j].bPoint.p0.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p0.z = msgCam60_1_Obj.objects[j].bPoint.p0.z + cam_coordinate_offset_Z;

    msgCam60_1_Obj.objects[j].bPoint.p1.x = msgCam60_1_Obj.objects[j].bPoint.p1.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p1.y = msgCam60_1_Obj.objects[j].bPoint.p1.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p1.z = msgCam60_1_Obj.objects[j].bPoint.p1.z + cam_coordinate_offset_Z;

    msgCam60_1_Obj.objects[j].bPoint.p2.x = msgCam60_1_Obj.objects[j].bPoint.p2.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p2.y = msgCam60_1_Obj.objects[j].bPoint.p2.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p2.z = msgCam60_1_Obj.objects[j].bPoint.p2.z + cam_coordinate_offset_Z;

    msgCam60_1_Obj.objects[j].bPoint.p3.x = msgCam60_1_Obj.objects[j].bPoint.p3.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p3.y = msgCam60_1_Obj.objects[j].bPoint.p3.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p3.z = msgCam60_1_Obj.objects[j].bPoint.p3.z + cam_coordinate_offset_Z;

    msgCam60_1_Obj.objects[j].bPoint.p4.x = msgCam60_1_Obj.objects[j].bPoint.p4.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p4.y = msgCam60_1_Obj.objects[j].bPoint.p4.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p4.z = msgCam60_1_Obj.objects[j].bPoint.p4.z + cam_coordinate_offset_Z;

    msgCam60_1_Obj.objects[j].bPoint.p5.x = msgCam60_1_Obj.objects[j].bPoint.p5.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p5.y = msgCam60_1_Obj.objects[j].bPoint.p5.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p5.z = msgCam60_1_Obj.objects[j].bPoint.p5.z + cam_coordinate_offset_Z;

    msgCam60_1_Obj.objects[j].bPoint.p6.x = msgCam60_1_Obj.objects[j].bPoint.p6.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p6.y = msgCam60_1_Obj.objects[j].bPoint.p6.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p6.z = msgCam60_1_Obj.objects[j].bPoint.p6.z + cam_coordinate_offset_Z;

    msgCam60_1_Obj.objects[j].bPoint.p7.x = msgCam60_1_Obj.objects[j].bPoint.p7.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p7.y = msgCam60_1_Obj.objects[j].bPoint.p7.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p7.z = msgCam60_1_Obj.objects[j].bPoint.p7.z + cam_coordinate_offset_Z;

    /************************************************************************/

    transform_coordinate_main(msgCam60_1_Obj.objects[j].cPoint, cam_coordinate_offset_X, cam_coordinate_offset_Y,
                              cam_coordinate_offset_Z);

    /************************************************************************/

    // ======>
    vDetectedObjectCAM_60_1.push_back(msgCam60_1_Obj.objects[j]);
  }

  Cam60_1_num_cb = 0;  // YF 2019081901
  Cam60_1_num = vDetectedObjectCAM_60_1.size();
  printf("vDetectedObjectCAM_60_1.size() = %ld \n", vDetectedObjectCAM_60_1.size());
  /************************************************************************/

  /************************************************************************/
  printf("Cam30_1_num_cb = %d \n", Cam30_1_num_cb);

  for (int j = 0; j < Cam30_1_num_cb; j++)
  {
    msgCam30_1_Obj.objects[j].header = msgCam30_1_Obj.header;  // add timestamp for cam30 object

    msgCam30_1_Obj.objects[j].bPoint.p0.x = msgCam30_1_Obj.objects[j].bPoint.p0.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p0.y = msgCam30_1_Obj.objects[j].bPoint.p0.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p0.z = msgCam30_1_Obj.objects[j].bPoint.p0.z + cam_coordinate_offset_Z;

    msgCam30_1_Obj.objects[j].bPoint.p1.x = msgCam30_1_Obj.objects[j].bPoint.p1.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p1.y = msgCam30_1_Obj.objects[j].bPoint.p1.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p1.z = msgCam30_1_Obj.objects[j].bPoint.p1.z + cam_coordinate_offset_Z;

    msgCam30_1_Obj.objects[j].bPoint.p2.x = msgCam30_1_Obj.objects[j].bPoint.p2.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p2.y = msgCam30_1_Obj.objects[j].bPoint.p2.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p2.z = msgCam30_1_Obj.objects[j].bPoint.p2.z + cam_coordinate_offset_Z;

    msgCam30_1_Obj.objects[j].bPoint.p3.x = msgCam30_1_Obj.objects[j].bPoint.p3.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p3.y = msgCam30_1_Obj.objects[j].bPoint.p3.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p3.z = msgCam30_1_Obj.objects[j].bPoint.p3.z + cam_coordinate_offset_Z;

    msgCam30_1_Obj.objects[j].bPoint.p4.x = msgCam30_1_Obj.objects[j].bPoint.p4.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p4.y = msgCam30_1_Obj.objects[j].bPoint.p4.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p4.z = msgCam30_1_Obj.objects[j].bPoint.p4.z + cam_coordinate_offset_Z;

    msgCam30_1_Obj.objects[j].bPoint.p5.x = msgCam30_1_Obj.objects[j].bPoint.p5.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p5.y = msgCam30_1_Obj.objects[j].bPoint.p5.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p5.z = msgCam30_1_Obj.objects[j].bPoint.p5.z + cam_coordinate_offset_Z;

    msgCam30_1_Obj.objects[j].bPoint.p6.x = msgCam30_1_Obj.objects[j].bPoint.p6.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p6.y = msgCam30_1_Obj.objects[j].bPoint.p6.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p6.z = msgCam30_1_Obj.objects[j].bPoint.p6.z + cam_coordinate_offset_Z;

    msgCam30_1_Obj.objects[j].bPoint.p7.x = msgCam30_1_Obj.objects[j].bPoint.p7.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p7.y = msgCam30_1_Obj.objects[j].bPoint.p7.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p7.z = msgCam30_1_Obj.objects[j].bPoint.p7.z + cam_coordinate_offset_Z;

    transform_coordinate_main(msgCam30_1_Obj.objects[j].cPoint, cam_coordinate_offset_X, cam_coordinate_offset_Y,
                              cam_coordinate_offset_Z);
    // ======>
    vDetectedObjectCAM_30_1.push_back(msgCam30_1_Obj.objects[j]);
  }

  Cam30_1_num_cb = 0;  // YF 2019081901
  Cam30_1_num = vDetectedObjectCAM_30_1.size();
  printf("vDetectedObjectCAM_30_1.size() = %ld \n", vDetectedObjectCAM_30_1.size());
  /************************************************************************/

  /************************************************************************/
  printf("Cam120_1_num_cb = %d \n", Cam120_1_num_cb);

  for (int j = 0; j < Cam120_1_num_cb; j++)
  {
    msgCam120_1_Obj.objects[j].header = msgCam120_1_Obj.header;  // add timestamp for cam120 object

    msgCam120_1_Obj.objects[j].bPoint.p0.x = msgCam120_1_Obj.objects[j].bPoint.p0.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p0.y = msgCam120_1_Obj.objects[j].bPoint.p0.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p0.z = msgCam120_1_Obj.objects[j].bPoint.p0.z + cam_coordinate_offset_Z;

    msgCam120_1_Obj.objects[j].bPoint.p1.x = msgCam120_1_Obj.objects[j].bPoint.p1.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p1.y = msgCam120_1_Obj.objects[j].bPoint.p1.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p1.z = msgCam120_1_Obj.objects[j].bPoint.p1.z + cam_coordinate_offset_Z;

    msgCam120_1_Obj.objects[j].bPoint.p2.x = msgCam120_1_Obj.objects[j].bPoint.p2.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p2.y = msgCam120_1_Obj.objects[j].bPoint.p2.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p2.z = msgCam120_1_Obj.objects[j].bPoint.p2.z + cam_coordinate_offset_Z;

    msgCam120_1_Obj.objects[j].bPoint.p3.x = msgCam120_1_Obj.objects[j].bPoint.p3.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p3.y = msgCam120_1_Obj.objects[j].bPoint.p3.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p3.z = msgCam120_1_Obj.objects[j].bPoint.p3.z + cam_coordinate_offset_Z;

    msgCam120_1_Obj.objects[j].bPoint.p4.x = msgCam120_1_Obj.objects[j].bPoint.p4.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p4.y = msgCam120_1_Obj.objects[j].bPoint.p4.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p4.z = msgCam120_1_Obj.objects[j].bPoint.p4.z + cam_coordinate_offset_Z;

    msgCam120_1_Obj.objects[j].bPoint.p5.x = msgCam120_1_Obj.objects[j].bPoint.p5.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p5.y = msgCam120_1_Obj.objects[j].bPoint.p5.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p5.z = msgCam120_1_Obj.objects[j].bPoint.p5.z + cam_coordinate_offset_Z;

    msgCam120_1_Obj.objects[j].bPoint.p6.x = msgCam120_1_Obj.objects[j].bPoint.p6.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p6.y = msgCam120_1_Obj.objects[j].bPoint.p6.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p6.z = msgCam120_1_Obj.objects[j].bPoint.p6.z + cam_coordinate_offset_Z;

    msgCam120_1_Obj.objects[j].bPoint.p7.x = msgCam120_1_Obj.objects[j].bPoint.p7.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p7.y = msgCam120_1_Obj.objects[j].bPoint.p7.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p7.z = msgCam120_1_Obj.objects[j].bPoint.p7.z + cam_coordinate_offset_Z;

    transform_coordinate_main(msgCam120_1_Obj.objects[j].cPoint, cam_coordinate_offset_X, cam_coordinate_offset_Y,
                              cam_coordinate_offset_Z);
    // ======>
    vDetectedObjectCAM_120_1.push_back(msgCam120_1_Obj.objects[j]);
  }

  Cam120_1_num_cb = 0;  // YF 2019081901
  Cam120_1_num = vDetectedObjectCAM_120_1.size();
  printf("vDetectedObjectCAM_120_1.size() = %ld \n", vDetectedObjectCAM_120_1.size());
  /************************************************************************/

  for (int j = 0; j < Lidar_num; j++)
  {
    if (msgLidarObj.objects[j].bPoint.p0.x > 0)
    {
      if (msgLidarObj.objects[j].bPoint.p0.x <= LID_Front_Short)
        vLidar_frontshort_Object.push_back(msgLidarObj.objects[j]);
      else
      {
        p0x = msgLidarObj.objects[j].bPoint.p0.x;
        p0y = msgLidarObj.objects[j].bPoint.p0.y;
        p3x = msgLidarObj.objects[j].bPoint.p3.x;
        p3y = msgLidarObj.objects[j].bPoint.p3.y;
        cx = (p0x + p3x) / 2;
        cy = (p0y + p3y) / 2;
        theta = atan2(cx, cy * (-1)) * (180 / PI);
        printf("theta :%f\n", theta);

        if (theta > 120)
          vLidar_60_0_Object.push_back(msgLidarObj.objects[j]);
        else if (theta < 60)
          vLidar_60_2_Object.push_back(msgLidarObj.objects[j]);
        else
          vLidar_60_1_Object.push_back(msgLidarObj.objects[j]);
      }
    }
    else
    {
      vLidar_rear_Object.push_back(msgLidarObj.objects[j]);  // rear
    }

    /*
    if((theta < 150) && (theta > 30))
     vLidar_120_2_Object.push_back(msgLidarObj.objects[j]);

    if((theta < 60) && (theta > -60))
     vLidar_120_1_Object.push_back(msgLidarObj.objects[j]);

    if((theta < -30) && (theta > -150))
     vLidar_120_0_Object.push_back(msgLidarObj.objects[j]);
    */
  }

  /************************************************************************/
  msgLidar_30_0_Obj.objects = vLidar_30_0_Object;
  msgLidar_30_1_Obj.objects = vLidar_30_1_Object;
  msgLidar_30_2_Obj.objects = vLidar_30_2_Object;

  msgLidar_60_0_Obj.objects = vLidar_60_0_Object;
  msgLidar_60_1_Obj.objects = vLidar_60_1_Object;
  msgLidar_60_2_Obj.objects = vLidar_60_2_Object;

  // msgLidar_120_0_Obj.objects = vLidar_120_0_Object;
  // msgLidar_120_1_Obj.objects = vLidar_120_1_Object;
  // msgLidar_120_2_Obj.objects = vLidar_120_2_Object;

  msgLidar_rear_Obj.objects = vLidar_rear_Object;
  msgLidar_frontshort.objects = vLidar_frontshort_Object;

  printf("put Lidar object to different view(total,60_0,60_1,60_2,30_1,rear) :%d,%ld,%ld,%ld,%ld\n", Lidar_num,
         msgLidar_60_0_Obj.objects.size(), msgLidar_60_1_Obj.objects.size(), msgLidar_60_2_Obj.objects.size(),
         msgLidar_rear_Obj.objects.size());

  printf("put Lidar object to different view(30_0,30_1,30_2) :%ld,%ld,%ld\n", msgLidar_30_0_Obj.objects.size(),
         msgLidar_30_1_Obj.objects.size(), msgLidar_30_2_Obj.objects.size());

  printf("put Lidar object to different view(120_0,120_1,120_2) :%ld,%ld,%ld\n", msgLidar_120_0_Obj.objects.size(),
         msgLidar_120_1_Obj.objects.size(), msgLidar_120_2_Obj.objects.size());
  /************************************************************************/

  Cam60_1_view_fusion();

  // Cam30_1_view_fusion();
  // Cam120_0_view_fusion();
  // Cam120_1_view_fusion();
  // Cam120_2_view_fusion();

  /************************************************************************/
  for (j = 0; j < vDetectedObjectCAM_120_1.size(); j++)
    vDetectedObjectDF.push_back(vDetectedObjectCAM_120_1[j]);

  for (j = 0; j < vDetectedObjectCAM_30_1.size(); j++)
    vDetectedObjectDF.push_back(vDetectedObjectCAM_30_1[j]);

  for (int j = 0; j < msgLidar_rear_Obj.objects.size(); j++)
    vDetectedObjectDF.push_back(msgLidar_rear_Obj.objects[j]);

  for (int j = 0; j < msgLidar_frontshort.objects.size(); j++)
    vDetectedObjectDF.push_back(msgLidar_frontshort.objects[j]);

  msgFusionObj.objects = vDetectedObjectDF;

  // add timestamp frame_id seq
  msgFusionObj.header.stamp = msgLidarObj.header.stamp;  // ros::Time::now();//msgLidarObj.header.stamp;
  msgFusionObj.header.frame_id = "SensorFusion";         // msgLidarObj.header.frame_id;
  msgFusionObj.header.seq = seq++;
  fusMsg_pub.publish(msgFusionObj);

  /************************************************************************/
}

void decision3DFusion()
{
  int tmp, a, b, c, d;
  int i, j;
  float p0x, p0y, p0z, p6x, p6y, p6z;
  float q0x, q0y, q0z, q6x, q6y, q6z;
  int case1, case2, skipRAD;
  int width_A = 0, width_B = 0, width_AB = 0, height_A = 0, height_B = 0, height_AB = 0;

  /************************************************************************/
  vDetectedObjectDF.clear();
  vDetectedObjectRAD.clear();
  vDetectedObjectLID.clear();
  vDetectedObjectCAM_60_1.clear();
  vDetectedObjectCAM_30_1.clear();
  vDetectedObjectCAM_120_1.clear();

  printf("Lidar_num_cb = %d \n", Lidar_num_cb);

  for (int j = 0; j < Lidar_num_cb; j++)
  {
    msgLidarObj.objects[j].header = msgLidarObj.header;  // add timestamp for lidar object

    msgLidarObj.objects[j].bPoint.p0.x = msgLidarObj.objects[j].bPoint.p0.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p0.y = msgLidarObj.objects[j].bPoint.p0.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p0.z = msgLidarObj.objects[j].bPoint.p0.z + lidar_coordinate_offset_Z;

    msgLidarObj.objects[j].bPoint.p1.x = msgLidarObj.objects[j].bPoint.p1.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p1.y = msgLidarObj.objects[j].bPoint.p1.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p1.z = msgLidarObj.objects[j].bPoint.p1.z + lidar_coordinate_offset_Z;

    msgLidarObj.objects[j].bPoint.p2.x = msgLidarObj.objects[j].bPoint.p2.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p2.y = msgLidarObj.objects[j].bPoint.p2.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p2.z = msgLidarObj.objects[j].bPoint.p2.z + lidar_coordinate_offset_Z;

    msgLidarObj.objects[j].bPoint.p3.x = msgLidarObj.objects[j].bPoint.p3.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p3.y = msgLidarObj.objects[j].bPoint.p3.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p3.z = msgLidarObj.objects[j].bPoint.p3.z + lidar_coordinate_offset_Z;

    msgLidarObj.objects[j].bPoint.p4.x = msgLidarObj.objects[j].bPoint.p4.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p4.y = msgLidarObj.objects[j].bPoint.p4.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p4.z = msgLidarObj.objects[j].bPoint.p4.z + lidar_coordinate_offset_Z;

    msgLidarObj.objects[j].bPoint.p5.x = msgLidarObj.objects[j].bPoint.p5.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p5.y = msgLidarObj.objects[j].bPoint.p5.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p5.z = msgLidarObj.objects[j].bPoint.p5.z + lidar_coordinate_offset_Z;

    msgLidarObj.objects[j].bPoint.p6.x = msgLidarObj.objects[j].bPoint.p6.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p6.y = msgLidarObj.objects[j].bPoint.p6.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p6.z = msgLidarObj.objects[j].bPoint.p6.z + lidar_coordinate_offset_Z;

    msgLidarObj.objects[j].bPoint.p7.x = msgLidarObj.objects[j].bPoint.p7.x + lidar_coordinate_offset_X;
    msgLidarObj.objects[j].bPoint.p7.y = msgLidarObj.objects[j].bPoint.p7.y + lidar_coordinate_offset_Y;
    msgLidarObj.objects[j].bPoint.p7.z = msgLidarObj.objects[j].bPoint.p7.z + lidar_coordinate_offset_Z;
    transform_coordinate_main(msgLidarObj.objects[j].cPoint, lidar_coordinate_offset_X, lidar_coordinate_offset_Y,
                              lidar_coordinate_offset_Z);
    // ======>
    vDetectedObjectLID.push_back(msgLidarObj.objects[j]);
  }

  Lidar_num = vDetectedObjectLID.size();
  printf("vDetectedObjectRAD.size() = %ld \n", vDetectedObjectLID.size());

  /************************************************************************/
  printf("radar_num_cb = %d \n", radar_num_cb);

  for (int j = 0; j < msgRadObj.objects.size(); j++)
  {
    msgRadObj.objects[j].header = msgRadObj.header;  // add timestamp for radar object

    msgRadObj.objects[j].bPoint.p0.x = msgRadObj.objects[j].bPoint.p0.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p0.y = msgRadObj.objects[j].bPoint.p0.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p0.z = msgRadObj.objects[j].bPoint.p0.z + radar_coordinate_offset_Z;

    msgRadObj.objects[j].bPoint.p1.x = msgRadObj.objects[j].bPoint.p1.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p1.y = msgRadObj.objects[j].bPoint.p1.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p1.z = msgRadObj.objects[j].bPoint.p1.z + radar_coordinate_offset_Z;

    msgRadObj.objects[j].bPoint.p2.x = msgRadObj.objects[j].bPoint.p2.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p2.y = msgRadObj.objects[j].bPoint.p2.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p2.z = msgRadObj.objects[j].bPoint.p2.z + radar_coordinate_offset_Z;

    msgRadObj.objects[j].bPoint.p3.x = msgRadObj.objects[j].bPoint.p3.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p3.y = msgRadObj.objects[j].bPoint.p3.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p3.z = msgRadObj.objects[j].bPoint.p3.z + radar_coordinate_offset_Z;

    msgRadObj.objects[j].bPoint.p4.x = msgRadObj.objects[j].bPoint.p4.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p4.y = msgRadObj.objects[j].bPoint.p4.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p4.z = msgRadObj.objects[j].bPoint.p4.z + radar_coordinate_offset_Z;

    msgRadObj.objects[j].bPoint.p5.x = msgRadObj.objects[j].bPoint.p5.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p5.y = msgRadObj.objects[j].bPoint.p5.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p5.z = msgRadObj.objects[j].bPoint.p5.z + radar_coordinate_offset_Z;

    msgRadObj.objects[j].bPoint.p6.x = msgRadObj.objects[j].bPoint.p6.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p6.y = msgRadObj.objects[j].bPoint.p6.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p6.z = msgRadObj.objects[j].bPoint.p6.z + radar_coordinate_offset_Z;

    msgRadObj.objects[j].bPoint.p7.x = msgRadObj.objects[j].bPoint.p7.x + radar_coordinate_offset_X;
    msgRadObj.objects[j].bPoint.p7.y = msgRadObj.objects[j].bPoint.p7.y + radar_coordinate_offset_Y;
    msgRadObj.objects[j].bPoint.p7.z = msgRadObj.objects[j].bPoint.p7.z + radar_coordinate_offset_Z;
    transform_coordinate_main(msgRadObj.objects[j].cPoint, radar_coordinate_offset_X, radar_coordinate_offset_Y,
                              radar_coordinate_offset_Z);
    // ======>
    vDetectedObjectRAD.push_back(msgRadObj.objects[j]);
  }

  radar_num = vDetectedObjectRAD.size();
  printf("vDetectedObjectRAD.size() = %ld \n", vDetectedObjectRAD.size());
  /************************************************************************/
  /************************************************************************/
  printf("Cam60_1_num_cb = %d \n", Cam60_1_num_cb);

  for (int j = 0; j < Cam60_1_num_cb; j++)
  {
    msgCam60_1_Obj.objects[j].header = msgCam60_1_Obj.header;  // add timestamp for cam object

    msgCam60_1_Obj.objects[j].bPoint.p0.x = msgCam60_1_Obj.objects[j].bPoint.p0.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p0.y = msgCam60_1_Obj.objects[j].bPoint.p0.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p0.z = msgCam60_1_Obj.objects[j].bPoint.p0.z + cam_coordinate_offset_Z;

    msgCam60_1_Obj.objects[j].bPoint.p1.x = msgCam60_1_Obj.objects[j].bPoint.p1.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p1.y = msgCam60_1_Obj.objects[j].bPoint.p1.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p1.z = msgCam60_1_Obj.objects[j].bPoint.p1.z + cam_coordinate_offset_Z;

    msgCam60_1_Obj.objects[j].bPoint.p2.x = msgCam60_1_Obj.objects[j].bPoint.p2.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p2.y = msgCam60_1_Obj.objects[j].bPoint.p2.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p2.z = msgCam60_1_Obj.objects[j].bPoint.p2.z + cam_coordinate_offset_Z;

    msgCam60_1_Obj.objects[j].bPoint.p3.x = msgCam60_1_Obj.objects[j].bPoint.p3.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p3.y = msgCam60_1_Obj.objects[j].bPoint.p3.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p3.z = msgCam60_1_Obj.objects[j].bPoint.p3.z + cam_coordinate_offset_Z;

    msgCam60_1_Obj.objects[j].bPoint.p4.x = msgCam60_1_Obj.objects[j].bPoint.p4.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p4.y = msgCam60_1_Obj.objects[j].bPoint.p4.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p4.z = msgCam60_1_Obj.objects[j].bPoint.p4.z + cam_coordinate_offset_Z;

    msgCam60_1_Obj.objects[j].bPoint.p5.x = msgCam60_1_Obj.objects[j].bPoint.p5.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p5.y = msgCam60_1_Obj.objects[j].bPoint.p5.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p5.z = msgCam60_1_Obj.objects[j].bPoint.p5.z + cam_coordinate_offset_Z;

    msgCam60_1_Obj.objects[j].bPoint.p6.x = msgCam60_1_Obj.objects[j].bPoint.p6.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p6.y = msgCam60_1_Obj.objects[j].bPoint.p6.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p6.z = msgCam60_1_Obj.objects[j].bPoint.p6.z + cam_coordinate_offset_Z;

    msgCam60_1_Obj.objects[j].bPoint.p7.x = msgCam60_1_Obj.objects[j].bPoint.p7.x + cam_coordinate_offset_X;
    msgCam60_1_Obj.objects[j].bPoint.p7.y = msgCam60_1_Obj.objects[j].bPoint.p7.y + cam_coordinate_offset_Y;
    msgCam60_1_Obj.objects[j].bPoint.p7.z = msgCam60_1_Obj.objects[j].bPoint.p7.z + cam_coordinate_offset_Z;
    transform_coordinate_main(msgCam60_1_Obj.objects[j].cPoint, cam_coordinate_offset_X, cam_coordinate_offset_Y,
                              cam_coordinate_offset_Z);
    // ======>
    vDetectedObjectCAM_60_1.push_back(msgCam60_1_Obj.objects[j]);
  }

  Cam60_1_num_cb = 0;  // YF 2019081901
  Cam60_1_num = vDetectedObjectCAM_60_1.size();
  printf("vDetectedObjectCAM_60_1.size() = %ld \n", vDetectedObjectCAM_60_1.size());
  /************************************************************************/
  /************************************************************************/
  printf("Cam30_1_num_cb = %d \n", Cam30_1_num_cb);

  for (int j = 0; j < Cam30_1_num_cb; j++)
  {
    msgCam30_1_Obj.objects[j].header = msgCam30_1_Obj.header;  // add timestamp for cam30 object

    msgCam30_1_Obj.objects[j].bPoint.p0.x = msgCam30_1_Obj.objects[j].bPoint.p0.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p0.y = msgCam30_1_Obj.objects[j].bPoint.p0.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p0.z = msgCam30_1_Obj.objects[j].bPoint.p0.z + cam_coordinate_offset_Z;

    msgCam30_1_Obj.objects[j].bPoint.p1.x = msgCam30_1_Obj.objects[j].bPoint.p1.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p1.y = msgCam30_1_Obj.objects[j].bPoint.p1.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p1.z = msgCam30_1_Obj.objects[j].bPoint.p1.z + cam_coordinate_offset_Z;

    msgCam30_1_Obj.objects[j].bPoint.p2.x = msgCam30_1_Obj.objects[j].bPoint.p2.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p2.y = msgCam30_1_Obj.objects[j].bPoint.p2.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p2.z = msgCam30_1_Obj.objects[j].bPoint.p2.z + cam_coordinate_offset_Z;

    msgCam30_1_Obj.objects[j].bPoint.p3.x = msgCam30_1_Obj.objects[j].bPoint.p3.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p3.y = msgCam30_1_Obj.objects[j].bPoint.p3.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p3.z = msgCam30_1_Obj.objects[j].bPoint.p3.z + cam_coordinate_offset_Z;

    msgCam30_1_Obj.objects[j].bPoint.p4.x = msgCam30_1_Obj.objects[j].bPoint.p4.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p4.y = msgCam30_1_Obj.objects[j].bPoint.p4.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p4.z = msgCam30_1_Obj.objects[j].bPoint.p4.z + cam_coordinate_offset_Z;

    msgCam30_1_Obj.objects[j].bPoint.p5.x = msgCam30_1_Obj.objects[j].bPoint.p5.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p5.y = msgCam30_1_Obj.objects[j].bPoint.p5.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p5.z = msgCam30_1_Obj.objects[j].bPoint.p5.z + cam_coordinate_offset_Z;

    msgCam30_1_Obj.objects[j].bPoint.p6.x = msgCam30_1_Obj.objects[j].bPoint.p6.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p6.y = msgCam30_1_Obj.objects[j].bPoint.p6.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p6.z = msgCam30_1_Obj.objects[j].bPoint.p6.z + cam_coordinate_offset_Z;

    msgCam30_1_Obj.objects[j].bPoint.p7.x = msgCam30_1_Obj.objects[j].bPoint.p7.x + cam_coordinate_offset_X;
    msgCam30_1_Obj.objects[j].bPoint.p7.y = msgCam30_1_Obj.objects[j].bPoint.p7.y + cam_coordinate_offset_Y;
    msgCam30_1_Obj.objects[j].bPoint.p7.z = msgCam30_1_Obj.objects[j].bPoint.p7.z + cam_coordinate_offset_Z;
    transform_coordinate_main(msgCam30_1_Obj.objects[j].cPoint, cam_coordinate_offset_X, cam_coordinate_offset_Y,
                              cam_coordinate_offset_Z);
    // ======>
    vDetectedObjectCAM_30_1.push_back(msgCam30_1_Obj.objects[j]);
  }

  Cam30_1_num_cb = 0;  // YF 2019081901
  Cam30_1_num = vDetectedObjectCAM_30_1.size();
  printf("vDetectedObjectCAM_30_1.size() = %ld \n", vDetectedObjectCAM_30_1.size());
  /************************************************************************/
  /************************************************************************/
  printf("Cam120_1_num_cb = %d \n", Cam120_1_num_cb);

  for (int j = 0; j < Cam120_1_num_cb; j++)
  {
    msgCam120_1_Obj.objects[j].header = msgCam120_1_Obj.header;  // add timestamp for cam120 object

    msgCam120_1_Obj.objects[j].bPoint.p0.x = msgCam120_1_Obj.objects[j].bPoint.p0.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p0.y = msgCam120_1_Obj.objects[j].bPoint.p0.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p0.z = msgCam120_1_Obj.objects[j].bPoint.p0.z + cam_coordinate_offset_Z;

    msgCam120_1_Obj.objects[j].bPoint.p1.x = msgCam120_1_Obj.objects[j].bPoint.p1.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p1.y = msgCam120_1_Obj.objects[j].bPoint.p1.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p1.z = msgCam120_1_Obj.objects[j].bPoint.p1.z + cam_coordinate_offset_Z;

    msgCam120_1_Obj.objects[j].bPoint.p2.x = msgCam120_1_Obj.objects[j].bPoint.p2.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p2.y = msgCam120_1_Obj.objects[j].bPoint.p2.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p2.z = msgCam120_1_Obj.objects[j].bPoint.p2.z + cam_coordinate_offset_Z;

    msgCam120_1_Obj.objects[j].bPoint.p3.x = msgCam120_1_Obj.objects[j].bPoint.p3.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p3.y = msgCam120_1_Obj.objects[j].bPoint.p3.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p3.z = msgCam120_1_Obj.objects[j].bPoint.p3.z + cam_coordinate_offset_Z;

    msgCam120_1_Obj.objects[j].bPoint.p4.x = msgCam120_1_Obj.objects[j].bPoint.p4.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p4.y = msgCam120_1_Obj.objects[j].bPoint.p4.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p4.z = msgCam120_1_Obj.objects[j].bPoint.p4.z + cam_coordinate_offset_Z;

    msgCam120_1_Obj.objects[j].bPoint.p5.x = msgCam120_1_Obj.objects[j].bPoint.p5.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p5.y = msgCam120_1_Obj.objects[j].bPoint.p5.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p5.z = msgCam120_1_Obj.objects[j].bPoint.p5.z + cam_coordinate_offset_Z;

    msgCam120_1_Obj.objects[j].bPoint.p6.x = msgCam120_1_Obj.objects[j].bPoint.p6.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p6.y = msgCam120_1_Obj.objects[j].bPoint.p6.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p6.z = msgCam120_1_Obj.objects[j].bPoint.p6.z + cam_coordinate_offset_Z;

    msgCam120_1_Obj.objects[j].bPoint.p7.x = msgCam120_1_Obj.objects[j].bPoint.p7.x + cam_coordinate_offset_X;
    msgCam120_1_Obj.objects[j].bPoint.p7.y = msgCam120_1_Obj.objects[j].bPoint.p7.y + cam_coordinate_offset_Y;
    msgCam120_1_Obj.objects[j].bPoint.p7.z = msgCam120_1_Obj.objects[j].bPoint.p7.z + cam_coordinate_offset_Z;
    transform_coordinate_main(msgCam120_1_Obj.objects[j].cPoint, cam_coordinate_offset_X, cam_coordinate_offset_Y,
                              cam_coordinate_offset_Z);
    // ======>
    vDetectedObjectCAM_120_1.push_back(msgCam120_1_Obj.objects[j]);
  }

  Cam120_1_num_cb = 0;  // YF 2019081901
  Cam120_1_num = vDetectedObjectCAM_120_1.size();
  printf("vDetectedObjectCAM_120_1.size() = %ld \n", vDetectedObjectCAM_120_1.size());
  /************************************************************************/

  for (j = 0; j < vDetectedObjectCAM_120_1.size(); j++)
    vDetectedObjectDF.push_back(vDetectedObjectCAM_120_1[j]);

  for (j = 0; j < vDetectedObjectCAM_30_1.size(); j++)
    vDetectedObjectDF.push_back(vDetectedObjectCAM_30_1[j]);

  for (j = 0; j < vDetectedObjectCAM_60_1.size(); j++)
    vDetectedObjectDF.push_back(vDetectedObjectCAM_60_1[j]);

  for (j = 0; j < vDetectedObjectLID.size(); j++)
    vDetectedObjectDF.push_back(vDetectedObjectLID[j]);

  case1 = 0;
  case2 = 0;
  skipRAD = 0;

  for (i = 0; i < vDetectedObjectRAD.size(); i++)
  {
    p0x = vDetectedObjectRAD[i].bPoint.p0.x;  // min
    p0y = vDetectedObjectRAD[i].bPoint.p0.y;  // min
    p0z = vDetectedObjectRAD[i].bPoint.p0.z;  // min
    p6x = vDetectedObjectRAD[i].bPoint.p6.x;  // max
    p6y = vDetectedObjectRAD[i].bPoint.p6.y;  // max
    p6z = vDetectedObjectRAD[i].bPoint.p6.z;  // max

    for (j = 0; j < vDetectedObjectLID.size(); j++)
    {
      q0x = vDetectedObjectLID[j].bPoint.p0.x;  // min
      q0y = vDetectedObjectLID[j].bPoint.p0.y;  // min
      q0z = vDetectedObjectLID[j].bPoint.p0.z;  // min
      q6x = vDetectedObjectLID[j].bPoint.p6.x;  // max
      q6y = vDetectedObjectLID[j].bPoint.p6.y;  // max
      q6z = vDetectedObjectLID[j].bPoint.p6.z;  // max

      // Known issues, fixed by below comment
      case1 = ((p0x < q6x) && (p0x > q0x)) && ((p0y > q6y) && (p0y < q0y));
      case2 = ((q0x < p6x) && (q0x > p0x)) && ((q0y > p6y) && (q0y < p0y));
      skipRAD = case1 || case2;
      // Corrected code
      //
      // width_A = p6x - p0x;
      // width_B = q6x - q0x;
      // width_AB = std::max(p6x, q6x) - std::min(p0x, q0x);
      // height_A = p6y - p0y;
      // height_B = q6y - q0y;
      // height_AB = std::max(p6y, q6y) - std::min(p0y, q0y);
      // skipRAD = (width_AB < width_A + width_B && height_AB < height_A + height_B);
    }

    // printf("###### skipRAD = %d #######\n",skipRAD);

    if (!skipRAD)  // only reserve non-overlap 3D Box from Radar
      vDetectedObjectDF.push_back(vDetectedObjectRAD[i]);
  }

  /************************************************************************/

  msgFusionObj.objects = vDetectedObjectDF;

  // add timestamp frame_id seq
  msgFusionObj.header.stamp = msgLidarObj.header.stamp;  // ros::Time::now();//msgLidarObj.header.stamp;
  msgFusionObj.header.frame_id = "SensorFusion";         // msgLidarObj.header.frame_id;
  msgFusionObj.header.seq = seq++;
  fusMsg_pub.publish(msgFusionObj);

  /************************************************************************/
}

void Cam60_0_view_fusion(void)
{
  int tmp, a, b, c, d;
  int i, j;
  double X_3d;
  double Y_3d;
  double Z_3d;
  int U_2d;
  int V_2d;
  int width_2d;
  int weight_2d;
  /************************************************************************/
  /*******************Cam60_0 view for decision fusion*********************/
  /************************************************************************/

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Camera-based Detection
  // int total_cam;

  for (i = 0; i < Cam60_0_num; i++)
  {
    cam_det[0][i] = Cam60_0_uv[i][0];
    cam_det[1][i] = Cam60_0_uv[i][1];
    cam_det[2][i] = Cam60_0_uv[i][2];
    cam_det[3][i] = Cam60_0_uv[i][3];
  }

  for (j = 0; j < max_det; j++)
  {
    cam_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(cam_det, Cam60_0_num);

  printf("**************************\n");
  for (j = 0; j < Cam60_0_num; j++)
    printf("1:cam60_0_Detection cam_det[][%d]  %d %d %d %d %d\n", j, cam_det[0][j], cam_det[1][j], cam_det[2][j],
           cam_det[3][j], cam_det[4][j]);

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Lidar-based Detection(Lidar_60_0)
  // int total_lidar;

  printf("msgLidar_60_0_Obj.objects.size() = %ld\n", msgLidar_60_0_Obj.objects.size());
  for (i = 0; i < msgLidar_60_0_Obj.objects.size(); i++)
  {
    Y_3d = msgLidar_60_0_Obj.objects[i].bPoint.p1.x - radar_coordinate_offset_X;
    X_3d = (msgLidar_60_0_Obj.objects[i].bPoint.p1.y) * (-1);
    Z_3d = msgLidar_60_0_Obj.objects[i].bPoint.p1.z;

    fusion_3D_to_2D(X_3d, Y_3d, Z_3d, &U_2d, &V_2d);

    lid_det[0][i] = U_2d;
    lid_det[1][i] = V_2d;

    printf("  %d,  %d \n", lid_det[0][i], lid_det[1][i]);

    Y_3d = msgLidar_60_0_Obj.objects[i].bPoint.p3.x - radar_coordinate_offset_X;
    X_3d = (msgLidar_60_0_Obj.objects[i].bPoint.p3.y) * (-1);
    Z_3d = msgLidar_60_0_Obj.objects[i].bPoint.p3.z;

    fusion_3D_to_2D(X_3d, Y_3d, Z_3d, &U_2d, &V_2d);

    width_2d = abs(U_2d - lid_det[0][i]);
    weight_2d = abs(V_2d - lid_det[1][i]);

    lid_det[2][i] = width_2d;
    lid_det[3][i] = weight_2d;
  }

  Lidar_num = msgLidar_60_0_Obj.objects.size();

  for (j = 0; j < max_det; j++)
  {
    lid_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(lid_det, Lidar_num);

  printf("**************************\n");
  for (j = 0; j < Lidar_num; j++)
    printf("0:LidarDetection lid_det[][%d]  %d %d %d %d %d\n", j, lid_det[0][j], lid_det[1][j], lid_det[2][j],
           lid_det[3][j], lid_det[4][j]);

  /**********************************************************************************************************/

  // Variables for Fused Detection

  overlap_fusion(cam_det, Cam60_0_num, lid_det, Lidar_num, bb_det, &total_det);
  total_det = Cam60_0_num + Lidar_num;

  printf("Lidar_num,Cam60_0_num,total_det => %d,%d,%d\n", Lidar_num, Cam60_0_num, total_det);

  /*********************************************************************************************************/
  for (int j = 0; j < Cam60_0_num; j++)
  {
    {
      if ((bb_det[0][j] == 0) || (bb_det[1][j] == 0))
        continue;

      if (bb_det[5][j] == 2)
      {
        vDetectedObjectDF.push_back(msgCam60_0_Obj.objects[j]);
      }
      else if (bb_det[5][j] == 0)
      {
        if (bb_det[4][j] == 0)
        {
          vDetectedObjectDF.push_back(msgCam60_0_Obj.objects[j]);
        }
        else if (bb_det[4][j] == 1)
        {
          vDetectedObjectDF.push_back(msgCam60_0_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;
  /************************************************************************/
  for (int j = 0; j < Lidar_num; j++)
  {
    {
      if ((bb_det[0][Cam60_0_num + j] == 0) || (bb_det[1][Cam60_0_num + j] == 0))
        continue;

      if (bb_det[5][Cam60_0_num + j] == 2)
      {
        vDetectedObjectDF.push_back(msgLidar_60_0_Obj.objects[j]);
      }
      else if (bb_det[5][Cam60_0_num + j] == 0)
      {
        if (bb_det[4][Cam60_0_num + j] == 0)
        {
          vDetectedObjectDF.push_back(msgLidar_60_0_Obj.objects[j]);
        }
        else if (bb_det[4][Cam60_0_num + j] == 1)
        {
          vDetectedObjectDF.push_back(msgLidar_60_0_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;

  /************************************************************************/

  /************************************************************************/
  /****************End Cam60_0 view for decision fusion********************/
  /************************************************************************/
}

void Cam60_1_view_fusion(void)
{
  float p0x, p0y, p0z, p6x, p6y, p6z;
  float q0x, q0y, q0z, q6x, q6y, q6z;
  float pxcenter, pycenter, qxcenter, qycenter;
  float diff;
  int ndet;
  int tmp, a, b, c, d;
  int i, j;
  double X_3d;
  double Y_3d;
  double Z_3d;
  int U_2d;
  int V_2d;
  int width_2d;
  int weight_2d;
  int check3D[max_det][max_det] = { 0 };
  /************************************************************************/
  /*******************Cam60_1 view for decision fusion*********************/
  /************************************************************************/

  int case1 = 0, width_A = 0, width_B = 0, width_AB = 0, height_A = 0, height_B = 0, height_AB = 0;

  for (i = 0; i < msgLidar_60_1_Obj.objects.size(); i++)
  {
    p0x = msgLidar_60_1_Obj.objects[i].bPoint.p0.x;  // min
    p0y = msgLidar_60_1_Obj.objects[i].bPoint.p0.y;  // min
    p6x = msgLidar_60_1_Obj.objects[i].bPoint.p6.x;  // max
    p6y = msgLidar_60_1_Obj.objects[i].bPoint.p6.y;  // max
    pxcenter = (p0x + p6x) / 2;
    pycenter = (p0y + p6y) / 2;

    for (j = 0; j < vDetectedObjectCAM_60_1.size(); j++)
    {
      q0x = vDetectedObjectCAM_60_1[j].bPoint.p0.x;  // min
      q0y = vDetectedObjectCAM_60_1[j].bPoint.p0.y;  // min
      q6x = vDetectedObjectCAM_60_1[j].bPoint.p6.x;  // max
      q6y = vDetectedObjectCAM_60_1[j].bPoint.p6.y;  // max
      qxcenter = (q0x + q6x) / 2;
      qycenter = (q0y + q6y) / 2;

      diff = sqrt((pxcenter - qxcenter) * (pxcenter - qxcenter) + (pycenter - qycenter) * (pycenter - qycenter));
      printf("distance is %f \n", diff);
      printf("pxcenter pycenter qxcenter qycenter is %f  %f  %f  %f \n", pxcenter, pycenter, qxcenter, qycenter);

      width_A = p6x - p0x;
      width_B = q6x - q0x;
      width_AB = std::max(p6x, q6x) - std::min(p0x, q0x);
      height_A = p6y - p0y;
      height_B = q6y - q0y;
      height_AB = std::max(p6y, q6y) - std::min(p0y, q0y);

      if ((diff < BB2BB_distance) || (width_AB < width_A + width_B && height_AB < height_A + height_B))
      {
        check3D[i][j] = 1;
        printf("Lidar %d and camera %d overlap \n", i, j);
      }
    }
  }

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Camera-based Detection
  // int total_cam;
  // int **cam_det;
  // cam_det = new int *[5];
  for (j = 0; j < 5; j++)
  {
    memset(cam_det[j], 0, sizeof(int) * max_det);
  }

  for (i = 0; i < Cam60_1_num; i++)
  {
    cam_det[0][i] = Cam60_1_uv[i][0];
    cam_det[1][i] = Cam60_1_uv[i][1];
    cam_det[2][i] = Cam60_1_uv[i][2];
    cam_det[3][i] = Cam60_1_uv[i][3];
  }

  for (j = 0; j < max_det; j++)
  {
    cam_det[4][j] = 0;
  }

  // mark for test
  // overlap_analysis(cam_det, Cam60_1_num);

  printf("**************************\n");
  for (j = 0; j < Cam60_1_num; j++)
    printf("1:cam60_1_Detection cam_det[][%d]  %d %d %d %d %d\n", j, cam_det[0][j], cam_det[1][j], cam_det[2][j],
           cam_det[3][j], cam_det[4][j]);

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Lidar-based Detection(Lidar_60_1)
  // int total_lidar;
  // int **lid_det;
  // lid_det = new int *[5];
  for (j = 0; j < 5; j++)
  {
    memset(lid_det[j], 0, sizeof(int) * max_det);
  }

  printf("msgLidar_60_1_Obj.objects.size() = %ld\n", msgLidar_60_1_Obj.objects.size());
  for (i = 0; i < msgLidar_60_1_Obj.objects.size(); i++)
  {
    Y_3d = msgLidar_60_1_Obj.objects[i].bPoint.p1.x - radar_coordinate_offset_X;
    X_3d = (msgLidar_60_1_Obj.objects[i].bPoint.p1.y) * (-1);
    Z_3d = msgLidar_60_1_Obj.objects[i].bPoint.p1.z;

    fusion_3D_to_2D(X_3d, Y_3d, Z_3d, &U_2d, &V_2d);

    lid_det[0][i] = U_2d;
    lid_det[1][i] = V_2d;

    printf("  %d,  %d \n", lid_det[0][i], lid_det[1][i]);

    Y_3d = msgLidar_60_1_Obj.objects[i].bPoint.p3.x - radar_coordinate_offset_X;
    X_3d = (msgLidar_60_1_Obj.objects[i].bPoint.p3.y) * (-1);
    Z_3d = msgLidar_60_1_Obj.objects[i].bPoint.p3.z;

    fusion_3D_to_2D(X_3d, Y_3d, Z_3d, &U_2d, &V_2d);

    width_2d = abs(U_2d - lid_det[0][i]);
    weight_2d = abs(V_2d - lid_det[1][i]);

    lid_det[2][i] = width_2d;
    lid_det[3][i] = weight_2d;
  }

  Lidar_num = msgLidar_60_1_Obj.objects.size();

  for (j = 0; j < max_det; j++)
  {
    lid_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  // overlap_analysis(lid_det, Lidar_num);

  printf("**************************\n");
  for (j = 0; j < Lidar_num; j++)
    printf("1:LidarDetection lid_det[][%d]  %d %d %d %d %d\n", j, lid_det[0][j], lid_det[1][j], lid_det[2][j],
           lid_det[3][j], lid_det[4][j]);

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for RaDAR-based Detection

  printf("msgRadObj.objects.size() = %ld\n", msgRadObj.objects.size());
  for (i = 0; i < msgRadObj.objects.size(); i++)
  {
    Y_3d = msgRadObj.objects[i].bPoint.p1.x - radar_coordinate_offset_X;
    X_3d = (msgRadObj.objects[i].bPoint.p1.y) * (-1);
    Z_3d = msgRadObj.objects[i].bPoint.p1.z;

    fusion_3D_to_2D(X_3d, Y_3d, Z_3d, &U_2d, &V_2d);

    radar_det[0][i] = U_2d;
    radar_det[1][i] = V_2d;

    printf("  %d,  %d \n", radar_det[0][i], radar_det[1][i]);

    Y_3d = msgRadObj.objects[i].bPoint.p3.x - radar_coordinate_offset_X;
    X_3d = (msgRadObj.objects[i].bPoint.p3.y) * (-1);
    Z_3d = msgRadObj.objects[i].bPoint.p3.z;

    fusion_3D_to_2D(X_3d, Y_3d, Z_3d, &U_2d, &V_2d);

    width_2d = abs(U_2d - radar_det[0][i]);
    weight_2d = abs(V_2d - radar_det[1][i]);

    radar_det[2][i] = width_2d;
    radar_det[3][i] = weight_2d;
  }

  for (j = 0; j < max_det; j++)
  {
    radar_det[4][j] = 0;
    // finalDETECT[j] = true;
  }

  // overlap_analysis(radar_det, radar_num);

  printf("**************************\n");
  for (j = 0; j < radar_num; j++)
    printf("1:RadarDetection radar_det[][%d]  %d %d %d %d %d\n", j, radar_det[0][j], radar_det[1][j], radar_det[2][j],
           radar_det[3][j], radar_det[4][j]);

  /**********************************************************************************************************/

  // Variables for Fused Detection
  // int total_det;
  // int **bb_det;
  // bb_det = new int *[6];
  for (int j = 0; j < 6; j++)
  {
    memset(bb_det[j], 0, sizeof(int) * (3 * max_det));
  }

  // Variables for Fused Detection
  // int total_det2;
  // int **bb_det2;
  // bb_det2 = new int *[6];
  for (int j = 0; j < 6; j++)
  {
    memset(bb_det2[j], 0, sizeof(int) * (3 * max_det));
  }

  /**********************************************************************************************************/
  // overlap_fusion(cam_det, Cam60_1_num,bb_det2, total_det2,  bb_det, &total_det);

  ndet = Cam60_1_num + Lidar_num;
  printf("ndet %d,Cam60_1_num %d,Lidar_num %d\n", ndet, Cam60_1_num, Lidar_num);

  for (int j = 0; j < 6; j++)
  {
    for (int i = 0; i < ndet; i++)
    {
      bb_det[j][i] = -1;
    }
  }
  ndet = 0;

  for (int j = 0; j < Cam60_1_num; j++)
  {
    {
      for (int i = 0; i < 4; i++)
      {
        bb_det[i][ndet] = cam_det[i][j];
      }
      bb_det[4][ndet] = 0;
      bb_det[5][ndet] = 0;
      ndet++;
    }
  }

  for (int j = 0; j < Lidar_num; j++)
  {
    {
      for (int i = 0; i < 4; i++)
      {
        bb_det[i][ndet] = lid_det[i][j];
      }
      bb_det[4][ndet] = 1;
      bb_det[5][ndet] = 0;
      ndet++;
    }
  }

  /**********************************************************************************************************/
  for (i = 0; i < Lidar_num; i++)
    for (j = 0; j < Cam60_1_num; j++)
      if (check3D[i][j] == 1)
      {
        /**********************************************************************************************************/
        // it can add the condition for choosing lidar or camera ?
        // is there any confidence for lidar or camera detection ?
        // is the object size fitting the vehcile?

        /**********************************************************************************************************/

        if (1)  // choose lidar
        {
          bb_det[4][j] = -1;
          bb_det[5][Cam60_1_num + i] = 2;
          bb_det[5][j] = -1;
        }
        else  // choose camera
        {
          bb_det[4][Cam60_1_num + i] = -1;
          bb_det[5][j] = 2;
          bb_det[5][Cam60_1_num + i] = -1;
        }
      }

  /**********************************************************************************************************/
  total_det = ndet;

  printf("**************overlap_fusion************\n");
  for (int j = 0; j < total_det; j++)
    printf("Cam60_1 view bb_det[][%d]  %d %d %d %d %d %d\n", j, bb_det[0][j], bb_det[1][j], bb_det[2][j], bb_det[3][j],
           bb_det[4][j], bb_det[5][j]);

  printf("**************overlap_fusion************\n");

  /*********************************************************************************************************/

  vDetectedObjectTemp.clear();

  for (int j = 0; j < Cam60_1_num; j++)
  {
    {
      if ((bb_det[0][j] == 0) || (bb_det[1][j] == 0))
        continue;

      if (bb_det[5][j] == 2)
      {
        vDetectedObjectDF.push_back(vDetectedObjectCAM_60_1[j]);
        vDetectedObjectTemp.push_back(vDetectedObjectCAM_60_1[j]);
      }
      else if (bb_det[5][j] == 0)
      {
        if (bb_det[4][j] == 0)
        {
          vDetectedObjectDF.push_back(vDetectedObjectCAM_60_1[j]);
          vDetectedObjectTemp.push_back(vDetectedObjectCAM_60_1[j]);
        }
        else if (bb_det[4][j] == 1)
        {
          vDetectedObjectDF.push_back(vDetectedObjectCAM_60_1[j]);
          vDetectedObjectTemp.push_back(vDetectedObjectCAM_60_1[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;
  /************************************************************************/
  for (int j = 0; j < Lidar_num; j++)
  {
    {
      if ((bb_det[0][Cam60_1_num + j] == 0) || (bb_det[1][Cam60_1_num + j] == 0))
        continue;

      if (bb_det[5][Cam60_1_num + j] == 2)
      {
        vDetectedObjectDF.push_back(msgLidar_60_1_Obj.objects[j]);
        vDetectedObjectTemp.push_back(msgLidar_60_1_Obj.objects[j]);
      }
      else if (bb_det[5][Cam60_1_num + j] == 0)
      {
        if (bb_det[4][Cam60_1_num + j] == 0)
        {
          vDetectedObjectDF.push_back(msgLidar_60_1_Obj.objects[j]);
          vDetectedObjectTemp.push_back(msgLidar_60_1_Obj.objects[j]);
        }
        else if (bb_det[4][Cam60_1_num + j] == 1)
        {
          vDetectedObjectDF.push_back(msgLidar_60_1_Obj.objects[j]);
          vDetectedObjectTemp.push_back(msgLidar_60_1_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;

  /************************************************************************/
  case1 = 0;
  for (i = 0; i < vDetectedObjectRAD.size(); i++)
  {
    p0x = vDetectedObjectRAD[i].bPoint.p0.x;  // min
    p0y = vDetectedObjectRAD[i].bPoint.p0.y;  // min
    p6x = vDetectedObjectRAD[i].bPoint.p6.x;  // max
    p6y = vDetectedObjectRAD[i].bPoint.p6.y;  // max
    pxcenter = (p0x + p6x) / 2;
    pycenter = (p0y + p6y) / 2;

    for (j = 0; j < vDetectedObjectTemp.size(); j++)
    {
      q0x = vDetectedObjectTemp[j].bPoint.p0.x;  // min
      q0y = vDetectedObjectTemp[j].bPoint.p0.y;  // min
      q6x = vDetectedObjectTemp[j].bPoint.p6.x;  // max
      q6y = vDetectedObjectTemp[j].bPoint.p6.y;  // max
      qxcenter = (q0x + q6x) / 2;
      qycenter = (q0y + q6y) / 2;

      diff = sqrt((pxcenter - qxcenter) * (pxcenter - qxcenter) + (pycenter - qycenter) * (pycenter - qycenter));
      printf("distance is %f \n", diff);
      printf("pxcenter pycenter qxcenter qycenter is %f  %f  %f  %f \n", pxcenter, pycenter, qxcenter, qycenter);

      width_A = p6x - p0x;
      width_B = q6x - q0x;
      width_AB = std::max(p6x, q6x) - std::min(p0x, q0x);
      height_A = p6y - p0y;
      height_B = q6y - q0y;
      height_AB = std::max(p6y, q6y) - std::min(p0y, q0y);

      case1 = (width_AB < width_A + width_B && height_AB < height_A + height_B) || (diff < BB2BB_distance);
    }

    if (case1 == 0)
      vDetectedObjectDF.push_back(vDetectedObjectRAD[i]);
  }

  /************************************************************************/

  /************************************************************************/
  /****************End Cam60_1 view for decision fusion********************/
  /************************************************************************/
}

void Cam60_2_view_fusion(void)
{
  int tmp, a, b, c, d;
  int i, j;
  double X_3d;
  double Y_3d;
  double Z_3d;
  int U_2d;
  int V_2d;
  int width_2d;
  int weight_2d;
  /************************************************************************/
  /*******************Cam60_2 view for decision fusion*********************/
  /************************************************************************/

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Camera-based Detection
  // int total_cam;
  // int **cam_det;
  // cam_det = new int *[5];
  for (j = 0; j < 5; j++)
  {
    memset(cam_det[j], 0, sizeof(int) * max_det);
  }

  for (i = 0; i < Cam60_2_num; i++)
  {
    cam_det[0][i] = Cam60_2_uv[i][0];
    cam_det[1][i] = Cam60_2_uv[i][1];
    cam_det[2][i] = Cam60_2_uv[i][2];
    cam_det[3][i] = Cam60_2_uv[i][3];
  }

  for (j = 0; j < max_det; j++)
  {
    cam_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(cam_det, Cam60_2_num);

  printf("**************************\n");
  for (j = 0; j < Cam60_2_num; j++)
    printf("1:cam60_2_Detection cam_det[][%d]  %d %d %d %d %d\n", j, cam_det[0][j], cam_det[1][j], cam_det[2][j],
           cam_det[3][j], cam_det[4][j]);

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Lidar-based Detection(Lidar_60_0)

  for (j = 0; j < 5; j++)
  {
    memset(lid_det[j], 0, sizeof(int) * max_det);
  }

  printf("msgLidar_60_2_Obj.objects.size() = %ld\n", msgLidar_60_2_Obj.objects.size());
  for (i = 0; i < msgLidar_60_2_Obj.objects.size(); i++)
  {
    Y_3d = msgLidar_60_2_Obj.objects[i].bPoint.p1.x - radar_coordinate_offset_X;
    X_3d = (msgLidar_60_2_Obj.objects[i].bPoint.p1.y) * (-1);
    Z_3d = msgLidar_60_2_Obj.objects[i].bPoint.p1.z;

    fusion_3D_to_2D(X_3d, Y_3d, Z_3d, &U_2d, &V_2d);

    lid_det[0][i] = U_2d;
    lid_det[1][i] = V_2d;

    printf("  %d,  %d \n", lid_det[0][i], lid_det[1][i]);

    Y_3d = msgLidar_60_2_Obj.objects[i].bPoint.p3.x - radar_coordinate_offset_X;
    X_3d = (msgLidar_60_2_Obj.objects[i].bPoint.p3.y) * (-1);
    Z_3d = msgLidar_60_2_Obj.objects[i].bPoint.p3.z;

    fusion_3D_to_2D(X_3d, Y_3d, Z_3d, &U_2d, &V_2d);

    width_2d = abs(U_2d - lid_det[0][i]);
    weight_2d = abs(V_2d - lid_det[1][i]);

    lid_det[2][i] = width_2d;
    lid_det[3][i] = weight_2d;
  }

  Lidar_num = msgLidar_60_2_Obj.objects.size();

  for (j = 0; j < max_det; j++)
  {
    lid_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(lid_det, Lidar_num);

  printf("**************************\n");
  for (j = 0; j < Lidar_num; j++)
    printf("2:LidarDetection lid_det[][%d]  %d %d %d %d %d\n", j, lid_det[0][j], lid_det[1][j], lid_det[2][j],
           lid_det[3][j], lid_det[4][j]);

  /**********************************************************************************************************/

  // Variables for Fused Detection
  // int total_det;
  // int **bb_det;
  // bb_det = new int *[6];
  for (int j = 0; j < 6; j++)
  {
    memset(bb_det[j], 0, sizeof(int) * (3 * max_det));
  }

  // Variables for Fused Detection
  // int total_det2;
  // int **bb_det2;
  // bb_det2 = new int *[6];
  for (int j = 0; j < 6; j++)
  {
    memset(bb_det2[j], 0, sizeof(int) * (3 * max_det));
  }

  overlap_fusion(cam_det, Cam60_2_num, lid_det, Lidar_num, bb_det, &total_det);
  total_det = Cam60_2_num + Lidar_num;

  printf("Lidar_num,Cam60_2_num,total_det => %d,%d,%d\n", Lidar_num, Cam60_2_num, total_det);

  /*********************************************************************************************************/
  for (int j = 0; j < Cam60_2_num; j++)
  {
    {
      if ((bb_det[0][j] == 0) || (bb_det[1][j] == 0))
        continue;

      if (bb_det[5][j] == 2)
      {
        vDetectedObjectDF.push_back(msgCam60_2_Obj.objects[j]);
      }
      else if (bb_det[5][j] == 0)
      {
        if (bb_det[4][j] == 0)
        {
          vDetectedObjectDF.push_back(msgCam60_2_Obj.objects[j]);
        }
        else if (bb_det[4][j] == 1)
        {
          vDetectedObjectDF.push_back(msgCam60_2_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;
  /************************************************************************/
  for (int j = 0; j < Lidar_num; j++)
  {
    {
      if ((bb_det[0][Cam60_2_num + j] == 0) || (bb_det[1][Cam60_2_num + j] == 0))
        continue;

      if (bb_det[5][Cam60_2_num + j] == 2)
      {
        vDetectedObjectDF.push_back(msgLidar_60_2_Obj.objects[j]);
      }
      else if (bb_det[5][Cam60_2_num + j] == 0)
      {
        if (bb_det[4][Cam60_2_num + j] == 0)
        {
          vDetectedObjectDF.push_back(msgLidar_60_2_Obj.objects[j]);
        }
        else if (bb_det[4][Cam60_2_num + j] == 1)
        {
          vDetectedObjectDF.push_back(msgLidar_60_2_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;

  /************************************************************************/

  /************************************************************************/
  /****************End Cam60_2 view for decision fusion********************/
  /************************************************************************/
}

void Cam30_0_view_fusion(void)
{
  int tmp, a, b, c, d;
  int i, j;

  /************************************************************************/
  /*******************Cam30_0 view for decision fusion*********************/
  /************************************************************************/

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Camera-based Detection
  // int total_cam;

  for (i = 0; i < Cam30_0_num; i++)
  {
    cam_det[0][i] = Cam30_0_uv[i][0];
    cam_det[1][i] = Cam30_0_uv[i][1];
    cam_det[2][i] = Cam30_0_uv[i][2];
    cam_det[3][i] = Cam30_0_uv[i][3];
  }

  for (j = 0; j < max_det; j++)
  {
    cam_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(cam_det, Cam30_0_num);

  printf("**************************\n");
  for (j = 0; j < Cam30_0_num; j++)
    printf("1:cam30_0_Detection cam_det[][%d]  %d %d %d %d %d\n", j, cam_det[0][j], cam_det[1][j], cam_det[2][j],
           cam_det[3][j], cam_det[4][j]);

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Lidar-based Detection(Lidar_30_0)
  // int total_lidar;

  for (i = 0; i < msgLidar_30_0_Obj.objects.size(); i++)
  {
    lid_det[0][i] = msgLidar_30_0_Obj.objects[i].lidarInfo.u;
    lid_det[1][i] = msgLidar_30_0_Obj.objects[i].lidarInfo.v;
    lid_det[2][i] = msgLidar_30_0_Obj.objects[i].lidarInfo.width;
    lid_det[3][i] = msgLidar_30_0_Obj.objects[i].lidarInfo.height;
  }

  Lidar_num = msgLidar_30_0_Obj.objects.size();

  for (j = 0; j < max_det; j++)
  {
    lid_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(lid_det, Lidar_num);

  printf("**************************\n");
  for (j = 0; j < Lidar_num; j++)
    printf("1:LidarDetection lid_det[][%d]  %d %d %d %d %d\n", j, lid_det[0][j], lid_det[1][j], lid_det[2][j],
           lid_det[3][j], lid_det[4][j]);

  /**********************************************************************************************************/

  // Variables for Fused Detection

  overlap_fusion(cam_det, Cam30_0_num, lid_det, Lidar_num, bb_det, &total_det);
  total_det = Cam30_0_num + Lidar_num;

  printf("Lidar_num,Cam30_0_num,total_det => %d,%d,%d\n", Lidar_num, Cam30_0_num, total_det);

  /*********************************************************************************************************/
  for (int j = 0; j < Cam30_0_num; j++)
  {
    {
      if ((bb_det[0][j] == 0) || (bb_det[1][j] == 0))
        continue;

      if (bb_det[5][j] == 2)
      {
        vDetectedObjectDF.push_back(msgCam30_0_Obj.objects[j]);
      }
      else if (bb_det[5][j] == 0)
      {
        if (bb_det[4][j] == 0)
        {
          vDetectedObjectDF.push_back(msgCam30_0_Obj.objects[j]);
        }
        else if (bb_det[4][j] == 1)
        {
          vDetectedObjectDF.push_back(msgCam30_0_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;
  /************************************************************************/
  for (int j = 0; j < Lidar_num; j++)
  {
    {
      if ((bb_det[0][Cam30_0_num + j] == 0) || (bb_det[1][Cam30_0_num + j] == 0))
        continue;

      if (bb_det[5][Cam30_0_num + j] == 2)
      {
        vDetectedObjectDF.push_back(msgLidar_30_0_Obj.objects[j]);
      }
      else if (bb_det[5][Cam30_0_num + j] == 0)
      {
        if (bb_det[4][Cam30_0_num + j] == 0)
        {
          vDetectedObjectDF.push_back(msgLidar_30_0_Obj.objects[j]);
        }
        else if (bb_det[4][Cam30_0_num + j] == 1)
        {
          vDetectedObjectDF.push_back(msgLidar_30_0_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;

  /************************************************************************/

  /************************************************************************/
  /****************End Cam30_0 view for decision fusion********************/
  /************************************************************************/
}

void Cam30_1_view_fusion(void)
{
  int tmp, a, b, c, d;
  int i, j;
  /************************************************************************/
  /*******************Cam30_1 view for decision fusion*********************/
  /************************************************************************/

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Camera-based Detection
  // int total_cam;
  // int **cam_det;
  // cam_det = new int *[5];
  for (j = 0; j < 5; j++)
  {
    memset(cam_det[j], 0, sizeof(int) * max_det);
  }

  for (i = 0; i < Cam30_1_num; i++)
  {
    cam_det[0][i] = Cam30_1_uv[i][0];
    cam_det[1][i] = Cam30_1_uv[i][1];
    cam_det[2][i] = Cam30_1_uv[i][2];
    cam_det[3][i] = Cam30_1_uv[i][3];
  }

  for (j = 0; j < max_det; j++)
  {
    cam_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(cam_det, Cam30_1_num);

  printf("**************************\n");
  for (j = 0; j < Cam30_1_num; j++)
    printf("1:cam30_1_Detection cam_det[][%d]  %d %d %d %d %d\n", j, cam_det[0][j], cam_det[1][j], cam_det[2][j],
           cam_det[3][j], cam_det[4][j]);

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Lidar-based Detection(Lidar_30_1)
  // int total_lidar;
  // int **lid_det;
  // lid_det = new int *[5];
  for (j = 0; j < 5; j++)
  {
    memset(lid_det[j], 0, sizeof(int) * max_det);
  }

  for (i = 0; i < msgLidar_30_1_Obj.objects.size(); i++)
  {
    lid_det[0][i] = msgLidar_30_1_Obj.objects[i].lidarInfo.u;
    lid_det[1][i] = msgLidar_30_1_Obj.objects[i].lidarInfo.v;
    lid_det[2][i] = msgLidar_30_1_Obj.objects[i].lidarInfo.width;
    lid_det[3][i] = msgLidar_30_1_Obj.objects[i].lidarInfo.height;
  }

  Lidar_num = msgLidar_30_1_Obj.objects.size();

  for (j = 0; j < max_det; j++)
  {
    lid_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(lid_det, Lidar_num);

  printf("**************************\n");
  for (j = 0; j < Lidar_num; j++)
    printf("1:LidarDetection lid_det[][%d]  %d %d %d %d %d\n", j, lid_det[0][j], lid_det[1][j], lid_det[2][j],
           lid_det[3][j], lid_det[4][j]);

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for RaDAR-based Detection
  // int total_radar;
  // int **radar_det;
  // radar_det = new int *[5];
  for (j = 0; j < 5; j++)
  {
    memset(radar_det[j], 0, sizeof(int) * max_det);
  }

  for (i = 0; i < radar_num; i++)
  {
    radar_det[0][i] = radar_uv[i][0];
    radar_det[1][i] = radar_uv[i][1];
    radar_det[2][i] = radar_uv[i][2];
    radar_det[3][i] = radar_uv[i][3];
  }

  for (j = 0; j < max_det; j++)
  {
    radar_det[4][j] = 0;
    // finalDETECT[j] = true;
  }

  overlap_analysis(radar_det, radar_num);

  printf("**************************\n");
  for (j = 0; j < radar_num; j++)
    printf("1:RadarDetection radar_det[][%d]  %d %d %d %d %d\n", j, radar_det[0][j], radar_det[1][j], radar_det[2][j],
           radar_det[3][j], radar_det[4][j]);

  /**********************************************************************************************************/

  // Variables for Fused Detection
  // int total_det;
  // int **bb_det;
  // bb_det = new int *[6];
  for (int j = 0; j < 6; j++)
  {
    memset(bb_det[j], 0, sizeof(int) * (3 * max_det));
  }

  // Variables for Fused Detection
  // int total_det2;
  // int **bb_det2;
  // bb_det2 = new int *[6];
  for (int j = 0; j < 6; j++)
  {
    memset(bb_det2[j], 0, sizeof(int) * (3 * max_det));
  }

  // overlap_fusion(cam_det, total_cam, radar_det, total_radar, bb_det, total_det);
  // bb_det <== cam_det + lid_det + radar_det
  // lidar and radar
  overlap_fusion(lid_det, Lidar_num, radar_det, radar_num, bb_det2, &total_det2);
  total_det2 = Lidar_num + radar_num;

  // 2nd overlap_fusion with lidar
  overlap_fusion(cam_det, Cam30_1_num, bb_det2, total_det2, bb_det, &total_det);
  total_det = total_det2 + Cam30_1_num;

  printf("Lidar_num,radar_num,Cam30_1_num,total_det => %d,%d,%d,%d\n", Lidar_num, radar_num, Cam30_1_num, total_det);

  /*********************************************************************************************************/
  for (int j = 0; j < Cam30_1_num; j++)
  {
    {
      if ((bb_det[0][j] == 0) || (bb_det[1][j] == 0))
        continue;

      if (bb_det[5][j] == 2)
      {
        vDetectedObjectDF.push_back(vDetectedObjectCAM_30_1[j]);
      }
      else if (bb_det[5][j] == 0)
      {
        if (bb_det[4][j] == 0)
        {
          vDetectedObjectDF.push_back(vDetectedObjectCAM_30_1[j]);
        }
        else if (bb_det[4][j] == 1)
        {
          vDetectedObjectDF.push_back(vDetectedObjectCAM_30_1[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;
  /************************************************************************/
  for (int j = 0; j < Lidar_num; j++)
  {
    {
      if ((bb_det[0][Cam30_1_num + j] == 0) || (bb_det[1][Cam30_1_num + j] == 0))
        continue;

      if (bb_det[5][Cam30_1_num + j] == 2)
      {
        vDetectedObjectDF.push_back(msgLidar_30_1_Obj.objects[j]);
      }
      else if (bb_det[5][Cam30_1_num + j] == 0)
      {
        if (bb_det[4][Cam30_1_num + j] == 0)
        {
          vDetectedObjectDF.push_back(msgLidar_30_1_Obj.objects[j]);
        }
        else if (bb_det[4][Cam30_1_num + j] == 1)
        {
          vDetectedObjectDF.push_back(msgLidar_30_1_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;

  /************************************************************************/
  for (int j = 0; j < radar_num; j++)
  {
    {
      if ((bb_det[0][Cam30_1_num + Lidar_num + j] == 0) || (bb_det[1][Cam30_1_num + Lidar_num + j] == 0))
        continue;

      if (bb_det[5][Cam30_1_num + Lidar_num + j] == 2)
      {
        vDetectedObjectDF.push_back(msgRadObj.objects[j]);
      }
      else if (bb_det[5][Cam30_1_num + Lidar_num + j] == 0)
      {
        if (bb_det[4][Cam30_1_num + Lidar_num + j] == 0)
        {
          vDetectedObjectDF.push_back(msgRadObj.objects[j]);
        }
        else if (bb_det[4][Cam30_1_num + Lidar_num + j] == 1)
        {
          vDetectedObjectDF.push_back(msgRadObj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;
  /************************************************************************/
  /****************End Cam30_1 view for decision fusion********************/
  /************************************************************************/
}

void Cam30_2_view_fusion(void)
{
  int tmp, a, b, c, d;
  int i, j;
  /************************************************************************/
  /*******************Cam30_2 view for decision fusion*********************/
  /************************************************************************/

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Camera-based Detection
  // int total_cam;
  // int **cam_det;
  // cam_det = new int *[5];
  for (j = 0; j < 5; j++)
  {
    memset(cam_det[j], 0, sizeof(int) * max_det);
  }

  for (i = 0; i < Cam30_2_num; i++)
  {
    cam_det[0][i] = Cam30_2_uv[i][0];
    cam_det[1][i] = Cam30_2_uv[i][1];
    cam_det[2][i] = Cam30_2_uv[i][2];
    cam_det[3][i] = Cam30_2_uv[i][3];
  }

  for (j = 0; j < max_det; j++)
  {
    cam_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(cam_det, Cam30_2_num);

  printf("**************************\n");
  for (j = 0; j < Cam30_2_num; j++)
    printf("1:cam30_2_Detection cam_det[][%d]  %d %d %d %d %d\n", j, cam_det[0][j], cam_det[1][j], cam_det[2][j],
           cam_det[3][j], cam_det[4][j]);

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Lidar-based Detection(Lidar_30_0)
  // int total_lidar;
  // int **lid_det;
  // lid_det = new int *[5];
  for (j = 0; j < 5; j++)
  {
    memset(lid_det[j], 0, sizeof(int) * max_det);
  }

  for (i = 0; i < msgLidar_30_2_Obj.objects.size(); i++)
  {
    lid_det[0][i] = msgLidar_30_2_Obj.objects[i].lidarInfo.u;
    lid_det[1][i] = msgLidar_30_2_Obj.objects[i].lidarInfo.v;
    lid_det[2][i] = msgLidar_30_2_Obj.objects[i].lidarInfo.width;
    lid_det[3][i] = msgLidar_30_2_Obj.objects[i].lidarInfo.height;
  }

  Lidar_num = msgLidar_30_2_Obj.objects.size();

  for (j = 0; j < max_det; j++)
  {
    lid_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(lid_det, Lidar_num);

  printf("**************************\n");
  for (j = 0; j < Lidar_num; j++)
    printf("1:LidarDetection lid_det[][%d]  %d %d %d %d %d\n", j, lid_det[0][j], lid_det[1][j], lid_det[2][j],
           lid_det[3][j], lid_det[4][j]);

  /**********************************************************************************************************/

  // Variables for Fused Detection
  // int total_det;
  // int **bb_det;
  // bb_det = new int *[6];
  for (int j = 0; j < 6; j++)
  {
    memset(bb_det[j], 0, sizeof(int) * (3 * max_det));
  }

  // Variables for Fused Detection
  // int total_det2;
  // int **bb_det2;
  // bb_det2 = new int *[6];
  for (int j = 0; j < 6; j++)
  {
    memset(bb_det2[j], 0, sizeof(int) * (3 * max_det));
  }

  overlap_fusion(cam_det, Cam30_2_num, lid_det, Lidar_num, bb_det, &total_det);
  total_det = Cam30_2_num + Lidar_num;

  printf("Lidar_num,Cam30_2_num,total_det => %d,%d,%d\n", Lidar_num, Cam30_2_num, total_det);

  /*********************************************************************************************************/
  for (int j = 0; j < Cam30_2_num; j++)
  {
    {
      if ((bb_det[0][j] == 0) || (bb_det[1][j] == 0))
        continue;

      if (bb_det[5][j] == 2)
      {
        vDetectedObjectDF.push_back(msgCam30_2_Obj.objects[j]);
      }
      else if (bb_det[5][j] == 0)
      {
        if (bb_det[4][j] == 0)
        {
          vDetectedObjectDF.push_back(msgCam30_2_Obj.objects[j]);
        }
        else if (bb_det[4][j] == 1)
        {
          vDetectedObjectDF.push_back(msgCam30_2_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;
  /************************************************************************/
  for (int j = 0; j < Lidar_num; j++)
  {
    {
      if ((bb_det[0][Cam30_2_num + j] == 0) || (bb_det[1][Cam30_2_num + j] == 0))
        continue;

      if (bb_det[5][Cam30_2_num + j] == 2)
      {
        vDetectedObjectDF.push_back(msgLidar_30_2_Obj.objects[j]);
      }
      else if (bb_det[5][Cam30_2_num + j] == 0)
      {
        if (bb_det[4][Cam30_2_num + j] == 0)
        {
          vDetectedObjectDF.push_back(msgLidar_30_2_Obj.objects[j]);
        }
        else if (bb_det[4][Cam30_2_num + j] == 1)
        {
          vDetectedObjectDF.push_back(msgLidar_30_2_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;

  /************************************************************************/

  /************************************************************************/
  /****************End Cam30_2 view for decision fusion********************/
  /************************************************************************/
}

void Cam120_0_view_fusion(void)
{
  int tmp, a, b, c, d;
  int i, j;

  /************************************************************************/
  /*******************Cam120_0 view for decision fusion*********************/
  /************************************************************************/

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Camera-based Detection
  // int total_cam;

  for (i = 0; i < Cam120_0_num; i++)
  {
    cam_det[0][i] = Cam120_0_uv[i][0];
    cam_det[1][i] = Cam120_0_uv[i][1];
    cam_det[2][i] = Cam120_0_uv[i][2];
    cam_det[3][i] = Cam120_0_uv[i][3];
  }

  for (j = 0; j < max_det; j++)
  {
    cam_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(cam_det, Cam120_0_num);

  printf("**************************\n");
  for (j = 0; j < Cam120_0_num; j++)
    printf("1:cam120_0_Detection cam_det[][%d]  %d %d %d %d %d\n", j, cam_det[0][j], cam_det[1][j], cam_det[2][j],
           cam_det[3][j], cam_det[4][j]);

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Lidar-based Detection(Lidar_120_0)
  // int total_lidar;

  for (i = 0; i < msgLidar_120_0_Obj.objects.size(); i++)
  {
    lid_det[0][i] = msgLidar_120_0_Obj.objects[i].lidarInfo.u;
    lid_det[1][i] = msgLidar_120_0_Obj.objects[i].lidarInfo.v;
    lid_det[2][i] = msgLidar_120_0_Obj.objects[i].lidarInfo.width;
    lid_det[3][i] = msgLidar_120_0_Obj.objects[i].lidarInfo.height;
  }

  Lidar_num = msgLidar_120_0_Obj.objects.size();

  for (j = 0; j < max_det; j++)
  {
    lid_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(lid_det, Lidar_num);

  printf("**************************\n");
  for (j = 0; j < Lidar_num; j++)
    printf("1:LidarDetection lid_det[][%d]  %d %d %d %d %d\n", j, lid_det[0][j], lid_det[1][j], lid_det[2][j],
           lid_det[3][j], lid_det[4][j]);

  /**********************************************************************************************************/

  // Variables for Fused Detection

  overlap_fusion(cam_det, Cam120_0_num, lid_det, Lidar_num, bb_det, &total_det);
  total_det = Cam120_0_num + Lidar_num;

  printf("Lidar_num,Cam120_0_num,total_det => %d,%d,%d\n", Lidar_num, Cam120_0_num, total_det);

  /*********************************************************************************************************/
  for (int j = 0; j < Cam120_0_num; j++)
  {
    {
      if ((bb_det[0][j] == 0) || (bb_det[1][j] == 0))
        continue;

      if (bb_det[5][j] == 2)
      {
        vDetectedObjectDF.push_back(msgCam120_0_Obj.objects[j]);
      }
      else if (bb_det[5][j] == 0)
      {
        if (bb_det[4][j] == 0)
        {
          vDetectedObjectDF.push_back(msgCam120_0_Obj.objects[j]);
        }
        else if (bb_det[4][j] == 1)
        {
          vDetectedObjectDF.push_back(msgCam120_0_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;
  /************************************************************************/
  for (int j = 0; j < Lidar_num; j++)
  {
    {
      if ((bb_det[0][Cam120_0_num + j] == 0) || (bb_det[1][Cam120_0_num + j] == 0))
        continue;

      if (bb_det[5][Cam120_0_num + j] == 2)
      {
        vDetectedObjectDF.push_back(msgLidar_120_0_Obj.objects[j]);
      }
      else if (bb_det[5][Cam120_0_num + j] == 0)
      {
        if (bb_det[4][Cam120_0_num + j] == 0)
        {
          vDetectedObjectDF.push_back(msgLidar_120_0_Obj.objects[j]);
        }
        else if (bb_det[4][Cam120_0_num + j] == 1)
        {
          vDetectedObjectDF.push_back(msgLidar_120_0_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;

  /************************************************************************/

  /************************************************************************/
  /****************End Cam120_0 view for decision fusion********************/
  /************************************************************************/
}

void Cam120_1_view_fusion(void)
{
  int tmp, a, b, c, d;
  int i, j;

  /************************************************************************/
  /*******************Cam120_1 view for decision fusion*********************/
  /************************************************************************/

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Camera-based Detection
  // int total_cam;

  for (i = 0; i < Cam120_1_num; i++)
  {
    cam_det[0][i] = Cam120_1_uv[i][0];
    cam_det[1][i] = Cam120_1_uv[i][1];
    cam_det[2][i] = Cam120_1_uv[i][2];
    cam_det[3][i] = Cam120_1_uv[i][3];
  }

  for (j = 0; j < max_det; j++)
  {
    cam_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(cam_det, Cam120_1_num);

  printf("**************************\n");
  for (j = 0; j < Cam120_1_num; j++)
    printf("1:cam120_1_Detection cam_det[][%d]  %d %d %d %d %d\n", j, cam_det[0][j], cam_det[1][j], cam_det[2][j],
           cam_det[3][j], cam_det[4][j]);

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Lidar-based Detection(Lidar_120_1)
  // int total_lidar;

  for (i = 0; i < msgLidar_120_1_Obj.objects.size(); i++)
  {
    lid_det[0][i] = msgLidar_120_1_Obj.objects[i].lidarInfo.u;
    lid_det[1][i] = msgLidar_120_1_Obj.objects[i].lidarInfo.v;
    lid_det[2][i] = msgLidar_120_1_Obj.objects[i].lidarInfo.width;
    lid_det[3][i] = msgLidar_120_1_Obj.objects[i].lidarInfo.height;
  }

  Lidar_num = msgLidar_120_1_Obj.objects.size();

  for (j = 0; j < max_det; j++)
  {
    lid_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(lid_det, Lidar_num);

  printf("**************************\n");
  for (j = 0; j < Lidar_num; j++)
    printf("1:LidarDetection lid_det[][%d]  %d %d %d %d %d\n", j, lid_det[0][j], lid_det[1][j], lid_det[2][j],
           lid_det[3][j], lid_det[4][j]);

  /**********************************************************************************************************/

  // Variables for Fused Detection

  overlap_fusion(cam_det, Cam120_1_num, lid_det, Lidar_num, bb_det, &total_det);
  total_det = Cam120_1_num + Lidar_num;

  printf("Lidar_num,Cam120_1_num,total_det => %d,%d,%d\n", Lidar_num, Cam120_1_num, total_det);

  /*********************************************************************************************************/
  for (int j = 0; j < Cam120_1_num; j++)
  {
    {
      if ((bb_det[0][j] == 0) || (bb_det[1][j] == 0))
        continue;

      if (bb_det[5][j] == 2)
      {
        vDetectedObjectDF.push_back(msgCam120_1_Obj.objects[j]);
      }
      else if (bb_det[5][j] == 0)
      {
        if (bb_det[4][j] == 0)
        {
          vDetectedObjectDF.push_back(msgCam120_1_Obj.objects[j]);
        }
        else if (bb_det[4][j] == 1)
        {
          vDetectedObjectDF.push_back(msgCam120_1_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;
  /************************************************************************/
  for (int j = 0; j < Lidar_num; j++)
  {
    {
      if ((bb_det[0][Cam120_1_num + j] == 0) || (bb_det[1][Cam120_1_num + j] == 0))
        continue;

      if (bb_det[5][Cam120_1_num + j] == 2)
      {
        vDetectedObjectDF.push_back(msgLidar_120_1_Obj.objects[j]);
      }
      else if (bb_det[5][Cam120_1_num + j] == 0)
      {
        if (bb_det[4][Cam120_1_num + j] == 0)
        {
          vDetectedObjectDF.push_back(msgLidar_120_1_Obj.objects[j]);
        }
        else if (bb_det[4][Cam120_1_num + j] == 1)
        {
          vDetectedObjectDF.push_back(msgLidar_120_1_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;

  /************************************************************************/

  /************************************************************************/
  /****************End Cam120_1 view for decision fusion********************/
  /************************************************************************/
}

void Cam120_2_view_fusion(void)
{
  int tmp, a, b, c, d;
  int i, j;
  /************************************************************************/
  /*******************Cam120_2 view for decision fusion*********************/
  /************************************************************************/

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Camera-based Detection
  // int total_cam;
  // int **cam_det;
  // cam_det = new int *[5];
  for (j = 0; j < 5; j++)
  {
    memset(cam_det[j], 0, sizeof(int) * max_det);
  }

  for (i = 0; i < Cam120_2_num; i++)
  {
    cam_det[0][i] = Cam120_2_uv[i][0];
    cam_det[1][i] = Cam120_2_uv[i][1];
    cam_det[2][i] = Cam120_2_uv[i][2];
    cam_det[3][i] = Cam120_2_uv[i][3];
  }

  for (j = 0; j < max_det; j++)
  {
    cam_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(cam_det, Cam120_2_num);

  printf("**************************\n");
  for (j = 0; j < Cam120_2_num; j++)
    printf("1:cam120_2_Detection cam_det[][%d]  %d %d %d %d %d\n", j, cam_det[0][j], cam_det[1][j], cam_det[2][j],
           cam_det[3][j], cam_det[4][j]);

  /************************************************************************/
  // prepare data for decision fusion
  // Variables for Lidar-based Detection(Lidar_120_2)
  // int total_lidar;
  // int **lid_det;
  // lid_det = new int *[5];
  for (j = 0; j < 5; j++)
  {
    memset(lid_det[j], 0, sizeof(int) * max_det);
  }

  for (i = 0; i < msgLidar_120_2_Obj.objects.size(); i++)
  {
    lid_det[0][i] = msgLidar_120_2_Obj.objects[i].lidarInfo.u;
    lid_det[1][i] = msgLidar_120_2_Obj.objects[i].lidarInfo.v;
    lid_det[2][i] = msgLidar_120_2_Obj.objects[i].lidarInfo.width;
    lid_det[3][i] = msgLidar_120_2_Obj.objects[i].lidarInfo.height;
  }

  Lidar_num = msgLidar_120_2_Obj.objects.size();

  for (j = 0; j < max_det; j++)
  {
    lid_det[4][j] = 0;
    // finalDETECT[j + max_det] = true;
  }

  overlap_analysis(lid_det, Lidar_num);

  printf("**************************\n");
  for (j = 0; j < Lidar_num; j++)
    printf("1:LidarDetection lid_det[][%d]  %d %d %d %d %d\n", j, lid_det[0][j], lid_det[1][j], lid_det[2][j],
           lid_det[3][j], lid_det[4][j]);

  /**********************************************************************************************************/

  // Variables for Fused Detection
  // int total_det;
  // int **bb_det;
  // bb_det = new int *[6];
  for (int j = 0; j < 6; j++)
  {
    memset(bb_det[j], 0, sizeof(int) * (3 * max_det));
  }

  // Variables for Fused Detection
  // int total_det2;
  // int **bb_det2;
  // bb_det2 = new int *[6];
  for (int j = 0; j < 6; j++)
  {
    memset(bb_det2[j], 0, sizeof(int) * (3 * max_det));
  }

  overlap_fusion(cam_det, Cam120_2_num, lid_det, Lidar_num, bb_det, &total_det);
  total_det = Cam120_2_num + Lidar_num;

  printf("Lidar_num,Cam120_2_num,total_det => %d,%d,%d\n", Lidar_num, Cam120_2_num, total_det);

  /*********************************************************************************************************/
  for (int j = 0; j < Cam120_2_num; j++)
  {
    {
      if ((bb_det[0][j] == 0) || (bb_det[1][j] == 0))
        continue;

      if (bb_det[5][j] == 2)
      {
        vDetectedObjectDF.push_back(msgCam120_2_Obj.objects[j]);
      }
      else if (bb_det[5][j] == 0)
      {
        if (bb_det[4][j] == 0)
        {
          vDetectedObjectDF.push_back(msgCam120_2_Obj.objects[j]);
        }
        else if (bb_det[4][j] == 1)
        {
          vDetectedObjectDF.push_back(msgCam120_2_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;
  /************************************************************************/
  for (int j = 0; j < Lidar_num; j++)
  {
    {
      if ((bb_det[0][Cam120_2_num + j] == 0) || (bb_det[1][Cam120_2_num + j] == 0))
        continue;

      if (bb_det[5][Cam120_2_num + j] == 2)
      {
        vDetectedObjectDF.push_back(msgLidar_120_2_Obj.objects[j]);
      }
      else if (bb_det[5][Cam120_2_num + j] == 0)
      {
        if (bb_det[4][Cam120_2_num + j] == 0)
        {
          vDetectedObjectDF.push_back(msgLidar_120_2_Obj.objects[j]);
        }
        else if (bb_det[4][Cam120_2_num + j] == 1)
        {
          vDetectedObjectDF.push_back(msgLidar_120_2_Obj.objects[j]);
        }
      }

      else
      {
      }
    }
  }  // cout << endl;

  /************************************************************************/

  /************************************************************************/
  /****************End Cam120_2 view for decision fusion********************/
  /************************************************************************/
}

void overlap_analysis(int** bb_det, int total_det)
{
  double** assoc;

  assoc = (double**)malloc(sizeof(double*) * max_det);
  for (int j = 0; j < max_det; j++)
  {
    assoc[j] = (double*)malloc(sizeof(double) * max_det);
    for (int i = 0; i < max_det; i++)
    {
      assoc[j][i] = 0;
    }
  }
  // Filter the bounding box
  int overlap_x, overlap_y, overlap, intersect, uni_area;
  double IOU;
  for (int j = 0; j < total_det; j++)
  {
    int x_scanA = bb_det[0][j], y_scanA = bb_det[1][j];
    int w_scanA = x_scanA + bb_det[2][j], h_scanA = y_scanA + bb_det[3][j];
    for (int i = 0; i < total_det; i++)
    {
      if (j != i)
      {
        // Calculate the IOU
        int x_scanB = bb_det[0][i], y_scanB = bb_det[1][i];
        int w_scanB = x_scanB + bb_det[2][i], h_scanB = y_scanB + bb_det[3][i];
        intersect = 0;
        for (int y = y_scanA; y < h_scanA; y++)
        {
          for (int x = x_scanA; x < w_scanA; x++)
          {
            overlap = 0;
            if (x >= x_scanB && x <= w_scanB)
            {
              overlap_x = 1;
            }
            else
            {
              overlap_x = 0;
            }
            if (y >= y_scanB && y <= h_scanB)
            {
              overlap_y = 1;
            }
            else
            {
              overlap_y = 0;
            }
            overlap = overlap_x * overlap_y;
            if (overlap == 1)
            {
              intersect++;
            }
          }
        }
        uni_area = (bb_det[2][i] * bb_det[3][i]) + (bb_det[2][j] * bb_det[3][j]) - intersect;
        if (uni_area == 0)
        {
          IOU = 0;
        }
        else
        {
          IOU = (double)intersect / (double)uni_area;
          // printf("IOU = %f\n",IOU);
        }
        assoc[j][i] = IOU;
      }
      else
      {
        assoc[j][i] = 0;
      }
    }
  }
  double bestnIOU;
  int bestnBB;
  for (int j = 0; j < total_det; j++)
  {
    for (int i = 0; i < total_det; i++)
    {
      if (assoc[j][i] > 0)
      {
        assoc[i][j] = 0;
        bestnIOU = assoc[j][i];
        bestnBB = i;
        for (int k = 0; k < total_det; k++)
        {
          IOU = assoc[i][k];
          if (IOU > 0)
          {
            if (bestnIOU < IOU)
            {
              assoc[bestnBB][k] = 0;
              assoc[k][bestnBB] = 0;
              bestnIOU = IOU;
              bestnBB = k;
            }
            else
            {
              assoc[i][k] = 0;
              assoc[k][i] = 0;
            }
          }
          else
          {
            // do nothing
          }
        }
      }
    }
  }
  int ncount;
  int mem1_, mem2_;
  bool toggle = false;
  for (int j = 0; j < total_det; j++)
  {
    ncount = 0;
    mem1_ = bb_det[4][j];
    toggle = false;
    if (mem1_ == 0)
    {
      for (int i = 0; i < total_det; i++)
      {
        mem2_ = bb_det[4][i];
        if (mem2_ == 0)
        {
          if (assoc[j][i] > 0)
          {
            // directly combine
            int startx_a = bb_det[0][j], starty_a = bb_det[1][j];
            int endx_a = startx_a + bb_det[2][j], endy_a = starty_a + bb_det[3][j];
            int startx_b = bb_det[0][i], starty_b = bb_det[1][i];
            int endx_b = startx_b + bb_det[2][i], endy_b = starty_b + bb_det[3][i];
            if (startx_a > startx_b)
            {
              bb_det[0][j] = startx_b;
              bb_det[0][i] = 0;
            }
            if (starty_a > starty_b)
            {
              bb_det[1][j] = starty_b;
              bb_det[1][i] = 0;
            }
            if (endx_a < endx_b)
            {
              bb_det[2][j] = endx_b - bb_det[0][j];
              bb_det[2][i] = 0;
            }
            else
            {
              bb_det[2][j] = endx_a - bb_det[0][j];
              bb_det[2][i] = 0;
            }
            if (endy_a < endy_b)
            {
              bb_det[3][j] = endy_b - bb_det[1][j];
              bb_det[3][i] = 0;
            }
            // Known issues, fixed by below comment
            if (endy_a < endy_b)
            {
              // Corrected code
              // if (endy_a > endy_b){
              bb_det[3][j] = endy_a - bb_det[1][j];
              bb_det[3][i] = 0;
            }
            toggle = true;
            bb_det[4][i] = -1;
          }
          else
          {
            ncount++;
          }
        }
        else
        {
        }
      }
      // Known issues, fixed by below comment
      if (ncount == total_det)
      {
        bb_det[4][j] = 1;
      }
      else
      {
        if (toggle == true)
        {
          bb_det[4][j] = 1;
        }
        else
        {
          bb_det[4][j] = 1;
        }
      }
      //
      // Corrected code
      // bb_det[4][j] = 1;
    }
  }

  for (int j = 0; j < max_det; j++)
    free(assoc[j]);

  free(assoc);
  printf("***********free memory 1**************\n");
}

// the function of overlap_fusion is to fuse detection results from different type of sensor with similar class
void overlap_fusion(int** cam, int ncam, int** rad, int nrad, int** det, int* total_det)
{
  double** assoc;

  int ndet;
  ndet = ncam + nrad;
  printf("ndet %d,ncam %d,nrad %d\n", ndet, ncam, nrad);

  for (int j = 0; j < 6; j++)
  {
    for (int i = 0; i < ndet; i++)
    {
      det[j][i] = -1;
    }
  }
  ndet = 0;
  for (int j = 0; j < ncam; j++)
  {
    // if (cam[4][j] == 1)
    {
      for (int i = 0; i < 4; i++)
      {
        det[i][ndet] = cam[i][j];
      }
      det[4][ndet] = 0;
      det[5][ndet] = 0;
      ndet++;
    }
  }
  for (int j = 0; j < nrad; j++)
  {
    // if (rad[4][j] == 1)
    {
      for (int i = 0; i < 4; i++)
      {
        det[i][ndet] = rad[i][j];
      }
      det[4][ndet] = 1;
      det[5][ndet] = 0;
      ndet++;
    }
  }
  assoc = (double**)malloc(sizeof(double*) * ndet);
  for (int j = 0; j < ndet; j++)
  {
    assoc[j] = (double*)malloc(sizeof(double) * ndet);
    for (int i = 0; i < ndet; i++)
    {
      assoc[j][i] = 0;
    }
  }

  // Filter the bounding box
  int overlap_x, overlap_y, overlap, intersect, uni_area;
  double IOU;
  for (int j = 0; j < ndet; j++)
  {
    int x_scanA = det[0][j], y_scanA = det[1][j];
    int w_scanA = x_scanA + det[2][j], h_scanA = y_scanA + det[3][j];
    for (int i = 0; i < ndet; i++)
    {
      if (j != i)
      {
        // Calculate the IOU
        int x_scanB = det[0][i], y_scanB = det[1][i];
        int w_scanB = x_scanB + det[2][i], h_scanB = y_scanB + det[3][i];
        intersect = 0;
        for (int y = y_scanA; y < h_scanA; y++)
        {
          for (int x = x_scanA; x < w_scanA; x++)
          {
            overlap = 0;
            if (x >= x_scanB && x <= w_scanB)
            {
              overlap_x = 1;
            }
            else
            {
              overlap_x = 0;
            }
            if (y >= y_scanB && y <= h_scanB)
            {
              overlap_y = 1;
            }
            else
            {
              overlap_y = 0;
            }
            overlap = overlap_x * overlap_y;
            if (overlap == 1)
            {
              intersect++;
            }
          }
        }
        uni_area = (det[2][i] * det[3][i]) + (det[2][j] * det[3][j]) - intersect;
        if (uni_area == 0)
        {
          IOU = 0;
        }
        else
        {
          IOU = (double)intersect / (double)uni_area;
        }
        assoc[j][i] = IOU;
      }
      else
      {
        assoc[j][i] = 0;
      }
    }
  }
  double bestnIOU;
  int bestnBB;
  for (int j = 0; j < ndet; j++)
  {
    for (int i = 0; i < ndet; i++)
    {
      if (assoc[j][i] > 0)
      {
        assoc[i][j] = 0;
        bestnIOU = assoc[j][i];
        bestnBB = i;
        for (int k = 0; k < ndet; k++)
        {
          IOU = assoc[i][k];
          if (IOU > 0)
          {
            if (bestnIOU < IOU)
            {
              assoc[bestnBB][k] = 0;
              assoc[k][bestnBB] = 0;
              bestnIOU = IOU;
              bestnBB = k;
            }
            else
            {
              assoc[i][k] = 0;
              assoc[k][i] = 0;
            }
          }
          else
          {
            // do nothing
          }
        }
      }
    }
  }

  for (int j = 0; j < ndet; j++)
  {
    int x_scanA = det[0][j], y_scanA = det[1][j];
    int w_scanA = x_scanA + det[2][j], h_scanA = y_scanA + det[3][j];
    for (int i = 0; i < ndet; i++)
    {
      if (assoc[j][i] > 0.150)
      {
        int startx_a = det[0][j], starty_a = det[1][j];
        int endx_a = startx_a + det[2][j], endy_a = starty_a + det[3][j];
        int startx_b = det[0][i], starty_b = det[1][i];
        int endx_b = startx_b + det[2][i], endy_b = starty_b + det[3][i];
        if (startx_a > startx_b)
        {
          det[0][j] = startx_b;
          det[0][i] = 0;
        }
        if (starty_a > starty_b)
        {
          det[1][j] = starty_b;
          det[1][i] = 0;
        }
        if (endx_a < endx_b)
        {
          det[2][j] = endx_b - det[0][j];
          det[2][i] = 0;
        }
        else
        {
          det[2][j] = endx_a - det[0][j];
          det[2][i] = 0;
        }
        if (endy_a < endy_b)
        {
          det[3][j] = endy_b - det[1][j];
          det[3][i] = 0;
        }
        if (endy_a < endy_b)
        {
          det[3][j] = endy_a - det[1][j];
          det[3][i] = 0;
        }

        det[4][i] = -1;
        det[5][j] = 2;
        det[5][i] = -1;
      }
      else if (assoc[j][i] > 0.000)
      {
        int startx_a = det[0][j], starty_a = det[1][j];
        int endx_a = startx_a + det[2][j], endy_a = starty_a + det[3][j];
        int startx_b = det[0][i], starty_b = det[1][i];
        int endx_b = startx_b + det[2][i], endy_b = starty_b + det[3][i];
        det[0][j] = (startx_a + startx_b) / 2;
        det[0][i] = 0;
        det[1][j] = (starty_a + starty_b) / 2;
        det[1][i] = 0;
        det[2][j] = ((endx_a + endx_b) / 2) - det[0][j];
        det[2][i] = 0;
        det[3][j] = ((endy_a + endy_b) / 2) - det[1][j];
        det[3][i] = 0;

        det[4][i] = -1;
        det[5][j] = 2;
        det[5][i] = -1;
      }
      else
      {
      }
    }
  }
  //*total = ndet;

  for (int j = 0; j < ndet; j++)
    free(assoc[j]);

  free(assoc);

  printf("***********free memory 2**************\n");
}

// QuickSort3 Algorithm
void swaps(int* a, int* b)
{
  int temp = *a;
  *a = *b;
  *b = temp;
}
void partition(int a[], int low, int high, int& i, int& j)
{
  if (high - low <= 1)
  {
    if (a[high] < a[low])
    {
      swaps(&a[high], &a[low]);
    }
    i = low;
    j = high;
    return;
  }
  int mid = low;
  int pivot = a[high];
  while (mid <= high)
  {
    if (a[mid] < pivot)
    {
      swaps(&a[low++], &a[mid++]);
    }
    else if (a[mid] == pivot)
    {
      mid++;
    }
    else if (a[mid] > pivot)
    {
      swaps(&a[mid], &a[high--]);
    }
  }
  i = low - 1;
  j = mid;
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

#ifdef EnableFusion

      decisionFusion();
      printf(" case1 \n");

#else

      decision3DFusion();
      printf(" case2 \n");

#endif

      // dbgPCView = 1;
      // pthread_mutex_lock(&mut_dbgPCView);
      // pthread_cond_wait(&cnd_dbgPCView, &mut_dbgPCView);
      // pthread_mutex_unlock(&mut_dbgPCView);

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

void* dbg_drawPointCloud(void* arg)
{
  // cv::namedWindow("BeforeFusion",CV_WINDOW_NORMAL);
  // cv::namedWindow("AfterFusion",CV_WINDOW_NORMAL);

  while (true)
  {
    if (dbgPCView != 1)
      continue;

    printf("************dbg_drawPointCloud***********\n");
    // cv::imshow("BeforeFusion", InImage);
    // cv::waitKey(1);
    // cv::imshow("AfterFusion", InImage2);
    // cv::waitKey(1);

    /*
                static uint32_t iteration = 0;
                static std::stringstream ss;
                ss<< std::setfill('0') << std::setw(6) << iteration << ".jpg";
                imwrite( (std::string("/home/user/code_hino/radar_fusion_sync/data/")+ss.str()).c_str(), InImage2 );
                ss.str(std::string());
                iteration++;
     */

    // cv::waitKey(1);
    pthread_mutex_lock(&mut_dbgPCView);
    dbgPCView = 0;
    pthread_cond_broadcast(&cnd_dbgPCView);
    pthread_mutex_unlock(&mut_dbgPCView);
  }  // end of while(true)

  // cv::destroyWindow("BeforeFusion");
  // cv::destroyWindow("AfterFusion");
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

int fusion_3D_to_2D(double x_3d, double y_3d, double z_3d, int* u_2d, int* v_2d)
{
  int i, j, k, u, v, tmp;
  int x_length, y_length;

  double radar_length;
  double radar_degree;
  double radar_x;
  double radar_y;
  int count = 0;

  // Object coordinate transformation

  radar_xyz[0] = x_3d;
  radar_xyz[1] = y_3d;
  radar_xyz[2] = -0.1;

  printf("radar_xyz[0],radar_xyz[1],radar_xyz[2] ==>  %f,  %f,  %f \n", radar_xyz[0], radar_xyz[1], radar_xyz[2]);

  matrix_vector_multiply_3x3_3d(R_Matrix, radar_xyz, temp);
  vector_add_3d(t_Vector, temp, camera_xyz);
  matrix_vector_multiply_3x3_3d(K_Matrix, camera_xyz, image_point);
  if (image_point[2] > 0)
  {
    u = (int)(image_point[0] / image_point[2]);
    v = (int)(image_point[1] / image_point[2]);

    printf("fov60: u is %d, v is %d \n", u, v);
  }

  if ((u > 0) && (u < 1920) && (v < 1208) && (v > 0))
  {
    if (u > (1920 - 11))
    {
      u = 1920 - 11;
    }
    else if (u < 10)
    {
      u = 10;
    }
    if (v > (1208 - 11))
    {
      v = 1208 - 11;
    }
    else if (v < 10)
    {
      v = 10;
    }

    *u_2d = u;
    *v_2d = v;
  }
  else
  {
    *u_2d = 0;
    *v_2d = 0;
  }

  printf("<== fusion_3D_to_2D \n");
}

void transform_coordinate_main(msgs::ConvexPoint& cp, const float x, const float y, const float z)
{
  for (unsigned i = 0; i < cp.lowerAreaPoints.size(); i++)
  {
    transform_coordinate(cp.lowerAreaPoints[i], x, y, z);
  }
}

void transform_coordinate(msgs::PointXYZ& p, const float x, const float y, const float z)
{
  p.x += x;
  p.y += y;
  p.z += z;
}