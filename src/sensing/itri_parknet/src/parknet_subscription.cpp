/*
   CREATER: ICL U300
   DATE: Feb, 2019
 */
#include "parknet_camera.h"
#include "parknet_subscription.h"
#include <string>
namespace parknet
{
const std::string subscription_topics_pcd[num_cams_e] = { "/LidarLeft", "/LidarFront", "/LidarRight" };

const std::string get_subscribed_image_topic(const int cam_id)
{
  auto port = get_camera_port();
  if (port.length() == 0)
  {
    return "/gmsl_camera/" + std::to_string(parknet::camera_id_mapping[cam_id]);
  }
  else
  {
    // Return a topic name like "/gmsl_camera/port_c/cam_0/image_raw"
    return "/gmsl_camera/port_" + port + "/cam_" + std::to_string(cam_id) + "/image_raw";
  }
}

};  // namespace parknet
