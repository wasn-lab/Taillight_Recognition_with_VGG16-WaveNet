#ifndef DECISION_FUSION
#define DECISION_FUSION

#include "ros/ros.h"
#include "std_msgs/String.h"
//#include <msgs/LidRoi.h>
//#include <msgs/CamObj.h>

#include <vector>

#define MAX_bbLID 50
#define MAX_bbCAM 50
#define MAX_bbMIX MAX_bbCAM

struct BBox2D {
    int ndxObj;
    int locX;
    int locY;
    int width;
    int height;
    float points[8][3];
    int klass;
    int id;
    float prob;
};

const static int FUS_WIDTH = 608;
const static int FUS_HEIGHT = 384;

extern std::vector<msgs::LidRoiBox> vLidRoi;
extern std::vector< BBox2D > vBbLidFront;
extern std::vector< BBox2D > vBbLidLeft;
extern std::vector< BBox2D > vBbLidRight;
extern std::vector< BBox2D > vBbLidOutofFOV;
extern std::vector< BBox2D > vBbRadFront;

extern std::vector<BBox2D> vBbCam10;
extern std::vector<BBox2D> vBbCam11;
extern std::vector<BBox2D> vBbCam12;

extern float ***bbCAMERA10;
extern float ***bbCAMERA11;
extern float ***bbCAMERA12;

extern bool *finaldetect10;
extern bool *finaldetect11;
extern bool *finaldetect12;

extern float **centroids10;
extern float **centroids11;
extern float **centroids12;

extern bool **association10;
extern bool **association11;
extern bool **association12;

extern bool **association_exclusive10;
extern bool **association_exclusive11;
extern bool **association_exclusive12;

extern int *camObjId10;
extern int *camObjId11;
extern int *camObjId12;
extern std_msgs::Header camHeader;

void LidObjCallback(const msgs::LidRoi::ConstPtr& msg);
void camBoxCb(const msgs::CamObj::ConstPtr& CamObj);
void decisionFusion();

void trans2dto3d10(std::vector<BBox2D>& bb2d, float*** bb3D, bool* finalDetect, int* camObjID, float* mtxR, float* mtxK, float* invR, float* mtxT, float* invK, int Numbblid);
void trans2dto3d11(std::vector<BBox2D>& bb2d, float*** bb3D, bool* finalDetect, int* camObjID, float* mtxR, float* mtxK, float* invR, float* mtxT, float* invK, int Numbblid);
void trans2dto3d12(std::vector<BBox2D>& bb2d, float*** bb3D, bool* finalDetect, int* camObjID, float* mtxR, float* mtxK, float* invR, float* mtxT, float* invK, int Numbblid);


#endif

