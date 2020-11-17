#include "xyz2lla.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "itri_xyz2lla");
  xyz2lla::XYZ2LLA app;
  app.run();
}
