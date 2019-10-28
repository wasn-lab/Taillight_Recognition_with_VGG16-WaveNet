/*
   CREATER: ICL U300
   DATE: May, 2019
 */

#ifndef __PARKNET_LOGGING_H__
#define __PARKNET_LOGGING_H__

#include <chrono>
#include "parknet.h"

#define DUMP_VALUE(x) LOG_INFO << #x << " = " << x << std::endl;

#define DURATION_VAR_NAME(name) _##name##Duration_count
#define TIME_IT_BEGIN(name) const auto _##name##Begin = std::chrono::high_resolution_clock::now();

#define TIME_IT_END(name)                                                                                              \
  const auto _##name##End = std::chrono::high_resolution_clock::now();                                                 \
  const auto _##name##Duration = _##name##End - _##name##Begin;                                                        \
  const auto DURATION_VAR_NAME(name) =                                                                                 \
      std::chrono::duration_cast<std::chrono::milliseconds>(_##name##Duration).count();                                \
  VLOG(2) << #name << " duration: " << DURATION_VAR_NAME(name) << "ms\n";

namespace parknet
{
int calc_duration_in_millisecond(std::chrono::time_point<std::chrono::high_resolution_clock> begin,
                                 std::chrono::time_point<std::chrono::high_resolution_clock> end);

};  // namespace parknet

#endif  // __PARKNET_LOGGING_H__
