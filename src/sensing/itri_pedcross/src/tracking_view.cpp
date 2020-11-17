#include "tracking_view.h"

namespace tra
{
void TrackingView::run()
{
  pedestrian_event();
}

void TrackingView::cache_image_callback(const sensor_msgs::Image::ConstPtr& msg)
{
#if PRINT_MESSAGE
  ros::Time start;
  start = ros::Time::now();
#endif

  // buffer raw image in cv::Mat with timestamp
  cv_bridge::CvImageConstPtr cv_ptr_image;
  cv_ptr_image = cv_bridge::toCvShare(msg, "bgr8");
  cv::Mat msg_decode;
  cv_ptr_image->image.copyTo(msg_decode);

  // buffer raw image in msg
  image_cache.push_back({ msg->header.stamp, msg_decode.clone() });
  msg_decode.release();
#if PRINT_MESSAGE
  std::cout << "Image buffer time cost: " << ros::Time::now() - start << std::endl;
#endif
}

void TrackingView::detection_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  latest_detection = *msg;
  draw_tracking_with_detection();
}
void TrackingView::tracking_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  latest_tracking = *msg;
  draw_tracking_with_detection();
}

void TrackingView::draw_tracking_with_detection()
{
  if (image_cache.empty())  // do if there is image in buffer
  {
    return;
  }

  cv::Mat matrix;
  cv::Mat matrix2;
  ros::Time msgs_timestamp = ros::Time(0);
  if (latest_detection.header.stamp.toSec() > latest_tracking.header.stamp.toSec())
  {
    msgs_timestamp = latest_detection.header.stamp;
  }
  else
  {
    msgs_timestamp = latest_tracking.header.stamp;
  }
  for (int i = image_cache.size() - 1; i >= 0; i--)
  {
    if (image_cache[i].first <= msgs_timestamp || i == 0)
    {
#if PRINT_MESSAGE
      std::cout << "GOT CHA !!!!! time: " << image_cache[i].first << " , " << msgs_timestamp << std::endl;
#endif

      matrix2 = image_cache[i].second;
      // for drawing bbox and keypoints
      matrix2.copyTo(matrix);
      // frame_timestamp = msgs_timestamp;
      break;
    }
  }

  // draw detection result
  for (const auto& obj : latest_detection.objects)
  {
    cv::Rect box;
    box.x = obj.camInfo[0].u * scaling_ratio_width;
    box.y = obj.camInfo[0].v * scaling_ratio_height;
    box.width = obj.camInfo[0].width * scaling_ratio_width;
    box.height = obj.camInfo[0].height * scaling_ratio_height;
    if (box.x + box.width > matrix.cols)
    {
      box.width = matrix.cols - box.x;
    }
    if (box.y + box.height > matrix.rows)
    {
      box.height = matrix.rows - box.y;
    }
    cv::rectangle(matrix, box.tl(), box.br(), CV_RGB(255, 255, 255), 2);
  }

  // draw tracking result
  for (const auto& obj : latest_tracking.objects)
  {
    cv::Rect box;
    box.x = obj.camInfo[0].u * scaling_ratio_width;
    box.y = obj.camInfo[0].v * scaling_ratio_height;
    box.width = obj.camInfo[0].width * scaling_ratio_width;
    box.height = obj.camInfo[0].height * scaling_ratio_height;
    if (box.x + box.width > matrix.cols)
    {
      box.width = matrix.cols - box.x;
    }
    if (box.y + box.height > matrix.rows)
    {
      box.height = matrix.rows - box.y;
    }
    cv::rectangle(matrix, box.tl(), box.br(), CV_RGB(0, 255, 0), 2);
    int tem_y = box.y;
    if (box.y >= 15)
    {
      box.y -= 15;
    }
    else
    {
      box.y = 0;
    }
    box.width = 30;
    box.height = 15;
    cv::rectangle(matrix, box.tl(), box.br(), CV_RGB(0, 255, 0), cv::FILLED);
    box.y = tem_y;
    std::string id_print = std::to_string(obj.track.id % 1000);
    cv::putText(matrix, id_print, box.tl(), cv::FONT_HERSHEY_PLAIN, 1 /*font size*/, cv::Scalar(0, 0, 0), 1, 2, false);
  }
  // make cv::Mat to sensor_msgs::Image
  sensor_msgs::ImageConstPtr viz_pub = cv_bridge::CvImage(std_msgs::Header(), "bgr8", matrix).toImageMsg();

  chatter_pub.publish(viz_pub);

  matrix.release();
  matrix2.release();
}

void TrackingView::pedestrian_event()
{
  // AsyncSpinner reference:
  //  https://gist.github.com/bgromov/45ebeced9e8067d9f13cceececc00d5b#file-test_spinner-cpp-L63

  // custom callback queue
  ros::CallbackQueue queue_1;
  ros::CallbackQueue queue_2;

  // This node handle uses global callback queue
  ros::NodeHandle nh_sub_1;
  // and this one uses custom queue
  ros::NodeHandle nh_sub_2;
  ros::NodeHandle nh_sub_3;

  // Set custom callback queue
  nh_sub_2.setCallbackQueue(&queue_1);
  nh_sub_3.setCallbackQueue(&queue_2);

  ros::Subscriber sub_1;
  ros::Subscriber sub_2;
  ros::Subscriber sub_3;

  sub_1 = nh_sub_1.subscribe("/cam_obj/front_bottom_60", 1, &TrackingView::detection_callback,
                             this);  // /Tracking2D/front_bottom_60 is subscirbe topic
  sub_2 = nh_sub_2.subscribe("/Tracking2D/front_bottom_60", 1, &TrackingView::tracking_callback,
                             this);  // /Tracking2D/left_back_60 is subscirbe topic
  sub_3 = nh_sub_3.subscribe("/cam/front_bottom_60", 1, &TrackingView::cache_image_callback,
                             this);  // /Tracking2D/left_back_60 is subscirbe topic

  // Create AsyncSpinner, run it on all available cores and make it process custom callback queue
  g_spinner_1.reset(new ros::AsyncSpinner(0, &queue_1));
  g_spinner_2.reset(new ros::AsyncSpinner(0, &queue_2));

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

      // Start the spinner
      g_spinner_1->start();
      g_spinner_2->start();

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

  // Wait for ROS threads to terminate
  ros::waitForShutdown();
}

}  // namespace tra
int main(int argc, char** argv)
{
#if USE_GLOG
  google::InstallFailureSignalHandler();
#endif
  ros::Time::init();
  ros::Time start, stop;
  start = ros::Time::now();
  ros::init(argc, argv, "tracking_view");

  tra::TrackingView tracking_view;
  ros::NodeHandle nh1;
  tracking_view.chatter_pub =
      nh1.advertise<sensor_msgs::Image&>("/TrackingView/front_bottom_60", 1);  // /PedCross/DrawBBox is pub topic
  tracking_view.image_cache = boost::circular_buffer<std::pair<ros::Time, cv::Mat>>(tracking_view.buffer_size);
  tracking_view.count = 0;

  stop = ros::Time::now();
  std::cout << "TrackingView started. Init time: " << stop - start << " sec" << std::endl;

  tracking_view.run();
  return 0;
}
