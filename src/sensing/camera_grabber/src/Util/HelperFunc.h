#ifndef __HEPLERFUNC__
#define __HEPLERFUNC__

#include <stdexcept>
#include <string>

namespace DEBUG
{
inline void NotImplementedExecption(const std::string& func_name)
{
  throw std::logic_error(func_name + "is not implemented yet.");
}

inline void OpenFileExecption(const std::string& video)
{
  throw std::logic_error("Can't not open video: " + video);
}
} // namespace DEBUG

namespace LOGGER
{
// Print performance of snychonized modules (usually used in demo mode)
inline void printSpeedInfoSync(float img, float lane, float signal, float drive60, float drive30, float opn, float all,
                               float noGPU_overhead_count = -1)
{
  // printf("\e[1;1H\e[2J");

  // Apear in Topic mode
  if (noGPU_overhead_count < 0)
  {
    printf("Camera Enabled 3 front 60 deg, 3 front 30 deg\n"
           "Modules run \"synchronously\":\n"
           "=============================================\n"
           "| All Frames Latancy  : %6.2f ms           \n"
           "| LaneNet Module      : %6.2f ms  %6.2f FPS\n"
           "| SignalNet Module    : %6.2f ms  %6.2f FPS\n"
           "| DriveNet 60 Module  : %6.2f ms  %6.2f FPS\n"
           "| DriveNet 30 Module  : %6.2f ms  %6.2f FPS\n"
           "| OpenRoadNet Module  : %6.2f ms  %6.2f FPS\n"
           "| All Done            : %6.2f ms  %6.2f FPS\n"
           "============================================\n"
           "\n\n\n",
           img, lane, 1000.0f / lane, signal, 1000.0f / signal, drive60, 1000.0f / drive60, drive30, 1000.0f / drive30,
           opn, 1000.0f / opn, all, 1000.0f / all);
  // Apear in demo mode
  }
  else
  {
    printf("Camera Enabled 3 front 60 deg, 3 front 30 deg\n"
           "Modules run \"synchronously\":\n"
           "=============================================\n"
           "| All Frames Latancy  : %6.2f ms           \n"
           "| LaneNet Module      : %6.2f ms  %6.2f FPS\n"
           "| SignalNet Module    : %6.2f ms  %6.2f FPS\n"
           "| DriveNet 60 Module  : %6.2f ms  %6.2f FPS\n"
           "| DriveNet 30 Module  : %6.2f ms  %6.2f FPS\n"
           "| OpenRoadNet Module  : %6.2f ms  %6.2f FPS\n"
           "| Non-gpu pre-process : %6.2f ms Overhead (DriveNet 60, DriveNet 30, OpenRoadNet)\n"
           "| All Done            : %6.2f ms  %6.2f FPS\n"
           "============================================\n"
           "\n\n\n",
           img, lane, 1000.0f / lane, signal, 1000.0f / signal, drive60, 1000.0f / drive60, drive30, 1000.0f / drive30,
           opn, 1000.0f / opn, noGPU_overhead_count, all, 1000.0f / all);
}
}
}  // namespace LOGGER

#endif
