#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
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
#include "TrtNet.h"
#include <string>
#include "dataReader.h"
#include <msgs/DetectedSignArray.h>
#include <msgs/DetectedLightArray.h>
#include "functions.h"
#include <opencv2/video/tracking_c.h>
#include "struct_define.h"

using namespace std;
//using namespace argsParser;
using namespace Tn;
using namespace Yolo;
using namespace cv;





std::unique_ptr<trtNet> net;
#ifdef ITRI_Field
string INPUT_PROTOTXT = "./416/DivU_TL_v1.prototxt";
string INPUT_CAFFEMODEL = "./416/DivU_TL_simplify.caffemodel";
string mode = "fp16";
RUN_MODE run_mode = RUN_MODE::FLOAT16;
string OUTPUTS= "yolo-det";//layer82-conv,layer94-conv,layer106-conv
#endif

#ifdef Tainan
const std::string INPUT_PROTOTXT = "yolov3_trt.prototxt";
const std::string INPUT_CAFFEMODEL = "yolov3.caffemodel";
string mode = "fp16"
RUN_MODE run_mode = RUN_MODE::FLOAT16;
string OUTPUTS= "yolo-det";//layer82-conv,layer94-conv,layer106-conv
#endif

int batchSize = 1;
string engineName = "./src/sensing/itri_lightnet/itri_lightnet_main/resources/LightNet.IPC_fp16.engine";
int outputCount;
unique_ptr<float[]> outputData(new float[100549]);
list<vector<Bbox>> outputs;


#ifdef ITRI_Field
int classNum = 2;
string class_name[2] = {"traffic_light", "bulb"};
#endif

#ifdef Tainan
int classNum = 105;
string class_name[105] = {"Maximum_Speed_30", "Maximum_Speed_40", "Maximum_Speed_50", "Maximum_Speed_60", "Maximum_Speed_70", "Maximum_Speed_80", "Maximum_Speed_90", "Maximum_Speed_100", "Maximum_Speed_110", "Minimum_Speed_30", "Minimum_Speed_40", "Minimum_Speed_50", "Minimum_Speed_60", "Minimum_Speed_70", "No_Left_Turn", "No_Right_Turn", "No_Left_Right_Turn", "No_Ahead_Left_Turn", "No_Ahead_Right_Turn", "No_U_Turn", "No_Pedestrian", "No_Bicycle_Motorbike", "No_Car", "No_Truck", "No_Car_Motorbike", "No_Overtake", "Overflow", "No_Parking", "No_Temporal_Parking", "No_Entry", "Workers_Ahead", "Road_Closed", "Right_Lane_Closed", "Left_Lane_Closed", "Detour", "One_Lane_Pass", "Follow_Sign", "Keep_Left", "Keep_Right", "Ahead_Only", "Left_Only", "Right_Only", "Left_Right_Only", "Ahead_Left_Only", "Ahead_Right_Only", "One_way", "infoRoundAb", "Yield", "Stop", "Pedestrian_Only", "Bicycle_Motorbike_Only", "Car_Only", "Bus_Only", "Left_Curve", "Right_Curve", "Curves_Right_Left", "Curves_Left_Right", "Uphill", "Downhill", "Road_Narrows", "Road_Narrows_on_Left", "Road_Narrows_on_Right", "Left_Merge", "Right_Merge", "Island_Ahead", "warnRoundAb", "Bi_direction", "Pedestrian_Crossing", "Children_Crossing", "Bicycle_Crossing", "Tunnel_Ahead", "Slow", "Danger", "SignBox", "red", "yellow", "green", "go_straight", "turn_right", "turn_left", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Traffic_light", "Warning_light", "back_light", "side_light", "pedestrian_light", "pedestrian_red", "pedestrian_green", "Child", "Adult", "Bicycle", "Motorcycle", "Bus", "Car", "Van", "Truck"};
#endif
vector<float> inputData;


//CV Post-process

int numOfDetection;
int tlClass;
Point tlStart, tlStop, tlStart2, tlStop2;
int numOfBody, numOfBulb;
int trafficLightBody[30][6];
int _tlBody[30][7];
//Information for _tlBody
//[0] = Inside TL Bulb Availability
//[1] = X Start
//[2] = Y Start
//[3] = X Stop
//[4] = Y Stop
//[5] = Dimension
//[6] = Depth
int trafficLightBulb[90][8];
int _tlBulb[90][9];
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
Point imageDimension;

int distance_light;
const int stateNum = 4;
const int measureNum = 2;
CvKalman *kalman;
CvMat *process_noise;
CvMat *measurement;
CvRNG rng;
float A[stateNum][stateNum] = { //transition matrix
	1, 0, 1, 0,
	0, 1, 0, 1,
	0, 0, 1, 0,
	0, 0, 0, 1};
int detect_count = 0;
#define detect_continuous_frame 5
int color_light = 0;
int direction = 0;


cv::Mat dImg;
Mat dImg1;