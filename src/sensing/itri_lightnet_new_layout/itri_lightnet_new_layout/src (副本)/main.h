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
#include <msgs/DetectedLightArray.h>
#include "functions.h"
#include <opencv2/video/tracking_c.h>
#include "struct_define.h"
#include <ros/package.h>


#define read_local_bagfile

using namespace std;
using namespace cv;

int camera_30deg_flag = 0,camera_60deg_flag = 0;
cv_bridge::CvImagePtr cv_ptr_30deg;
cv_bridge::CvImagePtr cv_ptr_60deg;

ros::Publisher Traffic_Light_pub;

//extern void DoNms(vector<Detection> &detections, int classes, float nmsThresh);
//extern vector<float> prepareImage(cv::Mat &img);
//extern vector<Bbox> postProcessImg(vector<Detection> &detections, int classes);
extern void DoNet(cv_bridge::CvImagePtr cv_ptr_30deg,cv_bridge::CvImagePtr cv_ptr_60deg, struct TL_status *TL_status_info, struct TL_color *TL_color_info);
extern void initi_all(const std::string& LightNet_TRT_model_path);
struct TL_status TL_status_info;
struct TL_color TL_color_info;