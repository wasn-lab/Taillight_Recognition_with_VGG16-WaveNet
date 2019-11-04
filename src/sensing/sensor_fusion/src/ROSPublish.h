#ifndef ROSPublish_H
#define ROSPublish_H
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <ros/ros.h>
#include <ros/package.h>
#include <msgs/DetectedObjectArray.h>
#include <msgs/DetectedObject.h>

class ROSPublish
{
private:
  const std::chrono::milliseconds interval;
  std::atomic_bool run;
  // ros::Publisher lanenet_pub;
  ros::Publisher fusMsg_pub;
  ros::NodeHandle nh;
  void RosTestData();

public:
  ROSPublish();
  ~ROSPublish();

  void stop();
  void tickFuntion();
  static void staticPublishCallbackFunction(void* p, msgs::DetectedObjectArray& msg)
  {
    // Get back into the class by treating p as the "this" pointer.
    ((ROSPublish*)p)->PublishcallbackFunction(msg);
  }
  void PublishcallbackFunction(msgs::DetectedObjectArray& msg);
};

#endif
