#ifndef PROJECTOR2_H
#define PROJECTOR2_H

#include <opencv2/opencv.hpp>
#include "parameters.h"
using namespace cv;
class Projector2 
{
  private:
    //當前的相機內外參
    CalibrateParameters currentParameters;
    //旋轉矩陣
    float R[9];
    //平移矩陣
    float T[3];
    //相機內參矩陣
    float K[9];

    void matrix_vector_multiply_3x3_3d(const float m[9], const float v[3],float result[3]);
    void vector_add_3d(const float v1[3], const float v2[3], float result[3]);
    void init_T_matrix();  
    void init_R_matrix();
    void init_K_matrix();

  public:
    static const int F_center = 0;
    static const int F_top = 1;
    static const int F_left = 2;
    static const int F_right = 3;
    static const int B_top = 4;
    static const int L_front = 5;
    static const int L_rear = 6;
    static const int R_front = 7;
    static const int R_rear = 8;
    
    static const int CALIBRATE_COMPLETED = F_left; 
   
    void init(int camera_id);
    vector<int> project(float x, float y, float z);
};
#endif
