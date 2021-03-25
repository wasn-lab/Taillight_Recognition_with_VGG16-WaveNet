#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include "opencv2/imgproc/imgproc.hpp"
#include <cv_bridge/cv_bridge.h>
#include <pthread.h>
#include <vector>
#include "sstream"
#include "fstream"
#include <iterator>
#include "dirent.h"
#include "iostream"
#include <chrono>
#include <stdlib.h>
//#include "TrtNet.h"
#include <string>
#include "dataReader.h"
#include <msgs/DetectedSignArray.h>
#include <msgs/DetectedLightArray.h>
#include "functions.h"
#include <opencv2/video/tracking_c.h>
#include "struct_define.h"
#include <stdio.h>
#include <time.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>
// #include "functions.h"


#include "class_timer.hpp"
#include "class_detector.h"

using namespace std;
//using namespace argsParser;
using namespace Tn;
//using namespace Yolo;
using namespace cv;
#define Tainan
//#define printf_debug

//std::unique_ptr<trtNet> net_30deg;
//std::unique_ptr<trtNet> net_60deg;
pthread_t thread_preprocess_30deg, thread_preprocess_60deg;
pthread_t thread_postprocess_30deg, thread_postprocess_60deg;
Mat animateFin;
Mat animate30;
Mat animate60;
/*
#ifdef ITRI_Field
string INPUT_PROTOTXT = "./416/DivU_TL_v1.prototxt";
string INPUT_CAFFEMODEL = "./416/DivU_TL_simplify.caffemodel";
string mode = "fp16";
RUN_MODE run_mode = RUN_MODE::FLOAT16;
string OUTPUTS = "yolo-det"; //layer82-conv,layer94-conv,layer106-conv
string engineName = "./src/sensing/itri_lightnet/itri_lightnet_main/resources/SignalNet.IPC_fp16.engine";
unique_ptr<float[]> outputData(new float[100549]);
#endif

#ifdef Tainan
const std::string INPUT_PROTOTXT_30deg = "./src/sensing/itri_lightnet_new_layout/itri_lightnet_new_layout/caffe_model/30deg/30_divU.prototxt";
const std::string INPUT_CAFFEMODEL_30deg = "./src/sensing/itri_lightnet_new_layout/itri_lightnet_new_layout/caffe_model/30deg/30_divU.caffemodel";
string mode_30deg = "fp16";
RUN_MODE run_mode_30deg = RUN_MODE::FLOAT32;
string OUTPUTS = "yolo-det"; //layer82-conv,layer94-conv,layer106-conv
string engineName_30deg = "/resources/yolov3_fp16_201208_30deg.engine";
unique_ptr<float[]> outputData_30deg(new float[402193]);

const std::string INPUT_PROTOTXT_60deg = "./src/sensing/itri_lightnet_new_layout/itri_lightnet_new_layout/caffe_model/60deg/60_divU.prototxt";
const std::string INPUT_CAFFEMODEL_60deg = "./src/sensing/itri_lightnet_new_layout/itri_lightnet_new_layout/caffe_model/60deg/60_divU.caffemodel";
string mode_60deg = "fp16";
RUN_MODE run_mode_60deg = RUN_MODE::FLOAT16;
string engineName_60deg = "/resources/yolov3_fp16_201208_60deg.engine";
unique_ptr<float[]> outputData_60deg(new float[402193]);
#endif
*/

int batchSize = 1;

int outputCount;
list<vector<Bbox>> outputs;
list<vector<Bbox>> outputs_60deg;

#ifdef ITRI_Field
int classNum = 2;
string class_name[2] = {"traffic_light", "bulb"};
#endif

#ifdef Tainan
int classNum = 6;
//string class_name[105] = {"Maximum_Speed_30", "Maximum_Speed_40", "Maximum_Speed_50", "Maximum_Speed_60", "Maximum_Speed_70", "Maximum_Speed_80", "Maximum_Speed_90", "Maximum_Speed_100", "Maximum_Speed_110", "Minimum_Speed_30", "Minimum_Speed_40", "Minimum_Speed_50", "Minimum_Speed_60", "Minimum_Speed_70", "No_Left_Turn", "No_Right_Turn", "No_Left_Right_Turn", "No_Ahead_Left_Turn", "No_Ahead_Right_Turn", "No_U_Turn", "No_Pedestrian", "No_Bicycle_Motorbike", "No_Car", "No_Truck", "No_Car_Motorbike", "No_Overtake", "Overflow", "No_Parking", "No_Temporal_Parking", "No_Entry", "Workers_Ahead", "Road_Closed", "Right_Lane_Closed", "Left_Lane_Closed", "Detour", "One_Lane_Pass", "Follow_Sign", "Keep_Left", "Keep_Right", "Ahead_Only", "Left_Only", "Right_Only", "Left_Right_Only", "Ahead_Left_Only", "Ahead_Right_Only", "One_way", "infoRoundAb", "Yield", "Stop", "Pedestrian_Only", "Bicycle_Motorbike_Only", "Car_Only", "Bus_Only", "Left_Curve", "Right_Curve", "Curves_Right_Left", "Curves_Left_Right", "Uphill", "Downhill", "Road_Narrows", "Road_Narrows_on_Left", "Road_Narrows_on_Right", "Left_Merge", "Right_Merge", "Island_Ahead", "warnRoundAb", "Bi_direction", "Pedestrian_Crossing", "Children_Crossing", "Bicycle_Crossing", "Tunnel_Ahead", "Slow", "Danger", "SignBox", "red", "yellow", "green", "go_straight", "turn_right", "turn_left", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Traffic_light", "Warning_light", "back_light", "side_light", "pedestrian_light", "pedestrian_red", "pedestrian_green", "Child", "Adult", "Bicycle", "Motorcycle", "Bus", "Car", "Van", "Truck"};
string class_name[7] = {"red", "yellow", "green", "go_straight", "turn_right", "turn_left", "Traffic_light"};

#endif
std::vector<Mat> inputData;
std::vector<Mat> inputData_60deg;

//CV Post-process

// int numOfDetection;
int numOfDetection_60deg;
int tlClass, normClass;
Point tlStart, tlStop, tlStart2, tlStop2;
int numOfBody, numOfBulb, selectTL, insideBulb[5];
int trafficLightBody[30][6], trafficLightBulb[90][8];
// int _tlBox30[90][10], numOfBox30;
// int _tlBox60[90][10], numOfBox60;

#ifdef ITRI_Field
int _tlBody[30][7];
//Information for _tlBody
//[0] = Inside TL Bulb Availability
//[1] = X Start
//[2] = Y Start
//[3] = X Stop
//[4] = Y Stop
//[5] = Dimension
//[6] = Depth
#endif
#ifdef Tainan
int selectTL30, selectTL60;
int _tlBox30[90][10], numOfBox30;
int _tlBox60[90][10], numOfBox60;
//Information for _tlBox
//[0] = Number of Bulb
//[1] = Depth
//[2-9] = 8 Slots for Traffic Light

int numOfDetection;
int trafficLightBulb30[90][8], numOfBulb30;
int _tlBulb30[90][10];
int trafficLightBulb60[90][8], numOfBulb60;
int _tlBulb60[90][10];
//Information for _tlBulb
//[0] = Availability Status
//[1] = X Start
//[2] = Y Start
//[3] = X Stop
//[4] = Y Stop
//[5] = Dimension
//[6] = Class
//[7] = Verified Class
//[8] = TL Box Position
//[9] = Depth Information
Point imageDimension;
bool preLightStatus30[6];
bool preLightStatus60[6];
int arrowPossibility[3];

bool IsItTheFirstFrame = true;
bool finLightStatus30[6];
int preLightCount30[6];
bool finLightStatus60[6];
int preLightCount60[6];
bool finLightStatus[2][6];
int finLightDistance[2];

#endif

int distance_light;
const int stateNum = 4;
const int measureNum = 2;
CvKalman *kalman;
CvMat *process_noise;
//CvMat *measurement;
Mat measurement;

CvRNG rng;
KalmanFilter KF(stateNum, measureNum, 0);
float A[stateNum][stateNum] = { //transition matrix
	1, 0, 10, 0,
	0, 1, 0, 10,
	0, 0, 1, 0,
	0, 0, 0, 1};
int detect_count = 0;
#define detect_continuous_frame 5
int color_light = 0;
int direction = 0;

cv::Mat dImg;
cv::Mat dImg_hdr;
Mat dImg1;

cv::Mat dImg_60deg;
cv::Mat dImg_60deg_hdr;
Mat dImg1_60deg;

int frame_count = 0;
Ptr<CLAHE> clahe_30deg;
Ptr<CLAHE> clahe_60deg;
Mat prediction ;


Config config_v4_60;
Config config_v4_30;

std::unique_ptr<Detector> detector_30(new Detector());
std::unique_ptr<Detector> detector_60(new Detector());

std::vector<BatchResult> batch_res_30;
std::vector<BatchResult> batch_res_60;

