#include "to_map_tf.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "itri_to_map_tf");
  to_map_tf::ToMapTF app;
  app.run();
}
