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

float min_ch(float R, float G, float B);
float max_ch(float R, float G, float B);
void adaptiveThreshold(int Body[9], Mat RGB, float &threshI, float &threshS);
int dimAspectRatio(int inpBulb[8]);
float intersectionOverUnion(int box1[8], int box2[8]);
void filterBB (int inpBulb[90][8], int &numInp, int outBulb[90][10], int &numOut, int camSelector);
void clusterBB (int Bulb[90][10], int numOut, int Box[90][10], int &numBox);
void statusRecognize30(int Bulb[90][10], int numBB, int Box[90][10], int numBox, int &selectedBox, Mat RGB, bool firstFrame, int preCount[6], bool finStat[6]);
void statusRecognize60(int Bulb[90][10], int numBB, int Box[90][10], int numBox, int &selectedBox, Mat RGB, bool firstFrame, int preCount[6], bool finStat[6]);
void animateTrafficLight(Mat animate, bool finStat[6], int depthTL);
void finalizeTrafficLight(bool StatA[6], int depthA, bool StatB[6], int depthB, bool finStat[2][6], int finDepth[2]);