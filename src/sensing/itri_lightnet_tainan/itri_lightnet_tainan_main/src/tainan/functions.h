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
void filterTrafficLight(int inp_tlBody[30][6], int &inp_numBody, int inp_tlBulb[90][8], int &inp_numBulb, 
						int out_tlBody[30][9], int out_tlBulb[90][9],
						int &selectBody, Point dimensions);
void adaptiveThreshold(int Body[9], Mat RGB, float &threshI, float &threshS);
int checkOrientation(int Body[9], int &minDim, int &maxDim);
float min_ch(float R, float G, float B);
float max_ch(float R, float G, float B);
void improveTrafficLight(int tlBody[9], int selectBody, int inBulb[5], int num_tlBulb, int tlBulb[90][9], Mat RGB);
void verifyTrafficLight(int tlBody[9], int inBulb[5], int tlBulb[90][9], Mat RGB);
void findTrafficLight(int tlBody[9], int selectBody, int inBulb[5], int &num_tlBulb, int tlBulb[90][9], Mat RGB);
void proposeTrafficLight(int tlBody[9], int selectBody, int inBulb[5], int &num_tlBulb, int tlBulb[90][9], Mat RGB);
void finalizeTrafficLight(int selectBody, int tlBody[30][9], int inBulb[5], int tlBulb[90][9], bool finStat[6], bool preStat[6]);