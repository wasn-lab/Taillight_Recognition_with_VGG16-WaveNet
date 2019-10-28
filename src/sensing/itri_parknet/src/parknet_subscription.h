/*
   CREATER: ICL U300
   DATE: Feb, 2019
 */

#ifndef __PARKNET_SUBSCRIPTION_H__
#define __PARKNET_SUBSCRIPTION_H__
#include <string>
#include "parknet_camera.h"
#include "parknet_args_parser.h"

namespace parknet
{
extern const std::string subscription_topics_pcd[num_cams_e];

const std::string get_subscribed_image_topic(const int cam_id);
};  // namespace parknet

#endif  // __PARKNET_SUBSCRIPTION_H__
