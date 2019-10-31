#include "CameraPerception.h"
#include "../Util/HelperFunc.h"

namespace SensingSubSystem
{
float CameraPerception::v_speed = .0f;
float CameraPerception::v_steeringAngle = .0f;
float CameraPerception::imu_angularZ = .0f;
float CameraPerception::imu_orientationX = .0f;
float CameraPerception::imu_orientationY = .0f;
float CameraPerception::imu_orientationZ = .0f;

CameraPerception::CameraPerception() : moduleName_("Basic Net")
{
}

CameraPerception::CameraPerception(const std::string moduleName) : moduleName_(moduleName)
{
}

CameraPerception::~CameraPerception()
{
}

void CameraPerception::initialize()
{
  // add your initialization steps
  //
  DEBUG::NotImplementedExecption(moduleName_ + "::CameraPerception::initialize");
}

void CameraPerception::run(const std::vector<void*>& raws_ptr)
{
  // main loop to run the perception
  //
  DEBUG::NotImplementedExecption(moduleName_ + "::CameraPerception::run cuda dptr");
}

void CameraPerception::display(std::vector<cv::Mat>& images)
{
  DEBUG::NotImplementedExecption(moduleName_ + "::CameraPerception::display mat");
}

void CameraPerception::run(std::vector<cv::Mat>& images)
{
  // main loop to run the perception
  //
  DEBUG::NotImplementedExecption(moduleName_ + "::CameraPerception::run mat");
}

void CameraPerception::release()
{
  // add your post steps when exiting the program
  //
  DEBUG::NotImplementedExecption(moduleName_ + "::CameraPerception::release");
}
}
