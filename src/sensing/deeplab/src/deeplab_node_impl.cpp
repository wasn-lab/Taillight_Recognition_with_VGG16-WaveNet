#include "deeplab_node_impl.h"
#include "deeplab_args_parser.h"
#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>

namespace deeplab {

static bool done_with_profiling()
{
  static int count = 0;
  if (!in_profiling_mode())
  {
    return false;
  }

  if (count > 10)
  {
    return true;
  }
  count++;
  return false;
}

DeeplabNodeImpl::DeeplabNodeImpl():
  segmenter_()
{
}

DeeplabNodeImpl::~DeeplabNodeImpl() = default;

void DeeplabNodeImpl::subscribe_topics()
{
  image_transport::ImageTransport it(node_handle_);
  image_subscriber_ = it.subscribe(get_image_topic(), 2, &DeeplabNodeImpl::image_callback, this);
}

void DeeplabNodeImpl::advertise_topics()
{
  auto topic = get_image_topic() + std::string("/segment");
  LOG(INFO) << "advertise " << topic;
  image_transport::ImageTransport it(node_handle_);
  image_publisher_ = it.advertise(topic, 1);
}

void DeeplabNodeImpl::image_callback(const sensor_msgs::ImageConstPtr& msg_in)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg_in, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    LOG(ERROR) << "cv_bridge exception: " << e.what();
    return;
  }
  cv::Mat img_out;
  segmenter_.segment(cv_ptr->image, img_out);

  LOG_EVERY_N(INFO, 67) << "Publish image";
  sensor_msgs::ImagePtr msg_out =
      cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::RGB8, img_out).toImageMsg();
  image_publisher_.publish(msg_out);
}

void DeeplabNodeImpl::run(int argc, char* argv[])
{
  subscribe_topics();
  advertise_topics();
  // ros::AsyncSpinner spinner(parknet::camera::num_cams_e);
  ros::AsyncSpinner spinner(1);
  spinner.start();
  ros::Rate r(30);
  while (ros::ok() && !done_with_profiling())
  {
    r.sleep();
  }
  spinner.stop();
  LOG(INFO) << "END detection";
}
}; //namespace deeplab
