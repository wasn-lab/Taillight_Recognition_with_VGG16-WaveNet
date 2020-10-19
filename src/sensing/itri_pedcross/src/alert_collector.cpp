#include "alert_collector.h"

namespace alr
{
void AlertCollector::run()
{
  alert_collector();
}

void AlertCollector::alert_front_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  alert_front = *msg;
  collect_and_publish();
}
void AlertCollector::alert_left_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  alert_left = *msg;
  collect_and_publish();
}
void AlertCollector::alert_right_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  alert_right = *msg;
  collect_and_publish();
}
void AlertCollector::alert_fov30_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  alert_fov30 = *msg;
  collect_and_publish();
}

void AlertCollector::collect_and_publish()
{
  msgs::DetectedObjectArray alert_array;
  alert_array.header.stamp = ros::Time(0);
  alert_array.header.frame_id = "base_link";
  for (auto const& obj : alert_front.objects)
  {
    alert_array.objects.emplace_back(obj);
  }
  for (auto const& obj : alert_left.objects)
  {
    alert_array.objects.emplace_back(obj);
  }
  for (auto const& obj : alert_right.objects)
  {
    alert_array.objects.emplace_back(obj);
  }
  for (auto const& obj : alert_fov30.objects)
  {
    alert_array.objects.emplace_back(obj);
  }
  chatter_pub.publish(alert_array);
}

void AlertCollector::alert_collector()
{
  // AsyncSpinner reference:
  //  https://gist.github.com/bgromov/45ebeced9e8067d9f13cceececc00d5b#file-test_spinner-cpp-L63

  // custom callback queue
  ros::CallbackQueue queue_1;
  ros::CallbackQueue queue_2;
  ros::CallbackQueue queue_3;

  // This node handle uses global callback queue
  ros::NodeHandle nh_sub_1;
  // and this one uses custom queue
  ros::NodeHandle nh_sub_2;
  ros::NodeHandle nh_sub_3;
  ros::NodeHandle nh_sub_4;

  // Set custom callback queue
  nh_sub_2.setCallbackQueue(&queue_1);
  nh_sub_3.setCallbackQueue(&queue_2);
  nh_sub_4.setCallbackQueue(&queue_3);

  ros::Subscriber sub_1;
  ros::Subscriber sub_2;
  ros::Subscriber sub_3;
  ros::Subscriber sub_4;

  sub_1 = nh_sub_1.subscribe("/PedCross/Alert/front_bottom_60", 1, &AlertCollector::alert_front_callback, this);
  sub_2 = nh_sub_2.subscribe("/PedCross/Alert/left_back_60", 1, &AlertCollector::alert_left_callback, this);
  sub_3 = nh_sub_3.subscribe("/PedCross/Alert/right_back_60", 1, &AlertCollector::alert_right_callback, this);
  sub_3 = nh_sub_4.subscribe("/PedCross/Alert/front_top_far_30", 1, &AlertCollector::alert_fov30_callback, this);

  // Create AsyncSpinner, run it on all available cores and make it process custom callback queue
  g_spinner_1.reset(new ros::AsyncSpinner(0, &queue_1));
  g_spinner_2.reset(new ros::AsyncSpinner(0, &queue_2));
  g_spinner_3.reset(new ros::AsyncSpinner(0, &queue_3));

  g_enable = true;
  g_trigger = true;

  // Loop with 100 Hz rate
  ros::Rate loop_rate(30);
  while (ros::ok())
  {
    // Enable state changed
    if (g_trigger)
    {
      // Clear old callback from the queue
      queue_1.clear();
      queue_2.clear();
      queue_3.clear();

      // Start the spinner
      g_spinner_1->start();
      g_spinner_2->start();
      g_spinner_3->start();

      ROS_INFO("Spinner enabled");
      // Reset trigger
      g_trigger = false;
    }

    // Process messages on global callback queue
    ros::spinOnce();
    loop_rate.sleep();
  }
  // Release AsyncSpinner object
  g_spinner_1.reset();
  g_spinner_2.reset();
  g_spinner_3.reset();

  // Wait for ROS threads to terminate
  ros::waitForShutdown();
}

}  // namespace alr
int main(int argc, char** argv)
{
#if USE_GLOG
  google::InstallFailureSignalHandler();
#endif
  ros::Time::init();
  ros::Time start, stop;
  start = ros::Time::now();
  ros::init(argc, argv, "alert_collector");

  alr::AlertCollector alert_collector;
  ros::NodeHandle nh1;
  alert_collector.chatter_pub =
      nh1.advertise<msgs::DetectedObjectArray>("/PedCross/Alert", 1);  // /PedCross/DrawBBox is pub topic
  alert_collector.count = 0;

  stop = ros::Time::now();
  std::cout << "AlertCollector started. Init time: " << stop - start << " sec" << std::endl;

  alert_collector.run();
  return 0;
}
