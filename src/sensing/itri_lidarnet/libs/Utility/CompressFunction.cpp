#include "CompressFunction.h"

CompressFunction::CompressFunction ()
{

}

CompressFunction::~CompressFunction ()
{

}

pcl::visualization::Camera
CompressFunction::CamPara (double A,
                           double B,
                           double C,
                           double D,
                           double E,
                           double F,
                           double G,
                           double H,
                           double I,
                           double J,
                           double K,
                           double L,
                           double M,
                           double N,
                           double O,
                           double P,
                           double Q)
{
  pcl::visualization::Camera cam;
  cam.pos[0] = A;
  cam.pos[1] = B;
  cam.pos[2] = C;
  cam.view[0] = D;
  cam.view[1] = E;
  cam.view[2] = F;
  cam.focal[0] = G;
  cam.focal[1] = H;
  cam.focal[2] = I;
  cam.clip[0] = J;
  cam.clip[1] = K;
  cam.clip[2] = L;
  cam.fovy = M;
  cam.window_pos[0] = N;
  cam.window_pos[1] = O;
  cam.window_size[0] = P;
  cam.window_size[1] = Q;
  return cam;
}

