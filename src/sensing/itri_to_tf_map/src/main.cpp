#include "to_tf_map.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "itri_to_tf_map");
  to_tf_map::ToTFMap app;
  app.run();
}
