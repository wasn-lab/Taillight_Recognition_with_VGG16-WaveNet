/*
   CREATER: ICL U300
   DATE: June, 2019
*/

#include <sys/stat.h>
#include "parknet.h"
#include "parknet_fs_utils.h"

namespace parknet
{
bool is_file(const std::string& name)
{
  struct stat buffer;
  if (stat(name.c_str(), &buffer) != 0)
  {
    return false;
  }
  return !S_ISDIR(buffer.st_mode);
}

std::string get_trt_engine_fullpath(const std::string& weights_file)
{
  return std::string(weights_file + ".tensorrt" + std::to_string(TENSORRT_VERSION_MAJOR) + "." +
                     std::to_string(TENSORRT_VERSION_MINOR) + "." + std::to_string(TENSORRT_VERSION_PATCH) + ".engine");
}

};  // namespace
