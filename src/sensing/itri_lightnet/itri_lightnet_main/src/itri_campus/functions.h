#include <string>
#include <stdlib.h>
#include <opencv2/highgui/highgui.hpp>  //for reading and visualizing image
#include <opencv2/imgproc/imgproc.hpp>  //for image resizing

//Defining PI value
#ifndef PI  
#ifdef M_PI
#define PI M_PI
#else
#define PI 3.14159265358979
#endif
#endif

using namespace cv;

//Data-type Converter Subroutine
std::string intToString(int number);
std::string floatToString(float number);
void filterTrafficLight(int inp_tlBody[30][6], int inp_numBody, int inp_tlBulb[90][8], int inp_numBulb, 
						int out_tlBody[30][7], int out_tlBulb[90][9],
						Point dimensions);
void confirmTrafficLight(int tlBody[30][7], int num_tlBody, int tlBulb[90][9], int num_tlBulb, Mat RGB);
void proposeTrafficLight(int tlBody[30][7], int num_tlBody, Mat RGB, int tlBulb[90][9], int &num_tlBulb);
void adaptiveThreshold(int Body[7], Mat RGB, float &threshI, float &threshS);
int checkOrientation(int Body[7], int &minDim, int &maxDim);
float min_ch(float R, float G, float B);
float max_ch(float R, float G, float B);