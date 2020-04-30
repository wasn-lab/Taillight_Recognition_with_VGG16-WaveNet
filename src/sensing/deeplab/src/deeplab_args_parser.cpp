#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <ros/package.h>
#include "deeplab_args_parser.h"

namespace deeplab
{
DEFINE_string(model_variant, "mobilenet_v2", "Model variant");
DEFINE_string(image_topic, "/cam/front_bottom_60", "Input image topic");
DEFINE_string(pb_file, "", "The pb file that is exported by deeplab model. If not set, then the default pb will be used.");

std::string get_image_topic()
{
  return FLAGS_image_topic;
}

std::string get_pb_file()
{
  if (FLAGS_pb_file.size() == 0)
  {
    return ros::package::getPath("deeplab") + "/models/frozen_inference_graph.pb";
  }
  return FLAGS_image_topic;
}


};  // namespace deeplab
