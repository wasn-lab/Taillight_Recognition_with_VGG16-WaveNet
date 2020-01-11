/*
   CREATER: ICL U300
   DATE: Feb, 2019
 */
#include "parknet_logging.h"
#include <chrono>
namespace parknet
{
bool do_logging()
{
  return true;
}

int calc_duration_in_millisecond(std::chrono::time_point<std::chrono::high_resolution_clock> begin,
                                 std::chrono::time_point<std::chrono::high_resolution_clock> end)
{
#error "test buildbot worker"
  auto dur = end - begin;
  return std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
}
};  // namespace parknet
