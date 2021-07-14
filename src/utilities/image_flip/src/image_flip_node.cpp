#include <thread>
#include <chrono>
#include <glog/logging.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Empty.h>
#include "image_flip_node.h"
#include "image_flip_args_parser.h"

namespace image_flip
{
ImageFlipNode::ImageFlipNode() = default;
ImageFlipNode::~ImageFlipNode() = default;

void ImageFlipNode::callback(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    LOG(ERROR) << "cv_bridge exception: " << e.what();
    return;
  }
  cv::Mat flip_img;
  cv::flip(cv_ptr->image, flip_img, 1);
  publisher_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", flip_img).toImageMsg());
  heartbeat_publisher_.publish(std_msgs::Empty());
}

static bool is_topic_published(const std::string& topic)
{
  ros::master::V_TopicInfo master_topics;
  ros::master::getTopics(master_topics);

  for (auto& master_topic : master_topics)
  {
    if (master_topic.name == topic)
    {
      return true;
    }
  }
  return false;
}

int ImageFlipNode::set_subscriber()
{
  std::string topic = image_flip::get_input_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty input topic name is not allow. Please pass it with -input_topic in the command line";
    return EXIT_FAILURE;
  }
  while (ros::ok() && !is_topic_published(topic))
  {
    LOG(INFO) << "wait 1 second for topic " << topic;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  LOG(INFO) << ros::this_node::getName() << ":"
            << " subscribe " << topic;
  subscriber_ = node_handle_.subscribe(topic, 2, &ImageFlipNode::callback, this);
  return EXIT_SUCCESS;
}

int ImageFlipNode::set_publisher()
{
  std::string topic = image_flip::get_output_topic();
  if (topic.empty())
  {
    LOG(ERROR) << "Empty output topic name is not allow. Please pass it with -output_topic in the command line";
    return EXIT_FAILURE;
  }
  LOG(INFO) << ros::this_node::getName() << ":"
            << " publish flipped image at topic " << topic;
  publisher_ = node_handle_.advertise<sensor_msgs::Image>(topic, /*queue size=*/2);
  heartbeat_publisher_ = node_handle_.advertise<std_msgs::Empty>(topic + "/heartbeat", /*queue size=*/1);
  return EXIT_SUCCESS;
}

void ImageFlipNode::run()
{
  CHECK(set_subscriber() == EXIT_SUCCESS);
  CHECK(set_publisher() == EXIT_SUCCESS);
  ros::AsyncSpinner spinner(/*thread_count*/1);
  spinner.start();
  ros::Rate r(0.1);
  while (ros::ok())
  {
    r.sleep();
  }
  spinner.stop();
}
};  // namespace image_flip
