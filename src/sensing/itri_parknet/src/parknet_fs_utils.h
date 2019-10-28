/*
   CREATER: ICL U300
   DATE: June, 2019
 */

#ifndef __PARKNET_FS_UTILS_H__
#define __PARKNET_FS_UTILS_H__
#include <string>

namespace parknet
{
bool is_file(const std::string& name);
std::string get_trt_engine_fullpath(const std::string& weights_file);
};  // namespace parknet

#endif  // __PARKNET_FS_UTILS_H__
