#include "RosImagePubSub.hpp"

ImageTransfer::ImageTransfer() : _mlock_data(new std::mutex())
{
  _data_ptr.reset(new cv::Mat);
  write_count = 0;
  read_count = 0;
}

bool ImageTransfer::put(const cv::Mat& content_in)
{
  // Lock
  //-------------------------------------------------------//
  std::lock_guard<std::mutex> _lock(*_mlock_data);
  //-------------------------------------------------------//
  // Fill the pointer
  if (!_data_ptr)
  {
    _data_ptr.reset(new cv::Mat);
  }
  // Copy element
  _copy_func(*_data_ptr, content_in);  // *ptr <-- data
  //-------------------//
  write_count++;
  return true;
}
bool ImageTransfer::put(std::shared_ptr<cv::Mat>& content_in_ptr)
{
  // Lock
  //-------------------------------------------------------//
  std::lock_guard<std::mutex> _lock(*_mlock_data);
  //-------------------------------------------------------//
  // The input is an empty pointer, return immediatly.
  if (!content_in_ptr)
  {
    return false;
  }
  // Fill the pointer
  if (!_data_ptr)
  {
    _data_ptr.reset(new cv::Mat);
  }
  if (!content_in_ptr.unique())
  {  // Not unique
    // Copy element
    _copy_func(*_data_ptr, *content_in_ptr);  // *ptr <-- *ptr
  }
  else
  {
    // The input pointer is pure (unique or null)
    // swapping pointers
    _data_ptr.swap(content_in_ptr);
    // Post-check: the output pointer should be pure (unique or empty)
    if (!content_in_ptr.unique())
    {                                     // Not not unique (empty or shared)
      content_in_ptr.reset(new cv::Mat);  // Reset the pointer to make it clean.
    }
    //
  }

  //-------------------//
  write_count++;
  return true;
}
bool ImageTransfer::get(std::shared_ptr<cv::Mat>& content_out_ptr)
{
  // Lock
  //-------------------------------------------------------//
  std::lock_guard<std::mutex> _lock(*_mlock_data);
  //-------------------------------------------------------//
  if (read_count == write_count)
  {
    return false;  // no new data
  }
  // Fill the pointer
  if (!_data_ptr)
  {
    _data_ptr.reset(new cv::Mat);
  }
  if (!content_out_ptr)
  {
    content_out_ptr.reset(new cv::Mat);
  }

  // Pre-check: The input pointer should be pure (unique or empty)
  if (is_ptr_shared(_data_ptr))
  {                                            // Not null and not unique
    _copy_func(*content_out_ptr, *_data_ptr);  // *ptr <-- *ptr
  }
  else
  {
    // The input pointer is pure (unique or null)
    // swapping
    content_out_ptr.swap(_data_ptr);
    // Post-check: the output pointer should be unique (not empty and not shared)
    _reset_ptr_if_shared(_data_ptr);
  }

  //-------------------//
  read_count = write_count;
  return true;
}

//==================================================//
//==================================================//

RosImagePubSub::RosImagePubSub(ros::NodeHandle& nh_in) : _nh_ptr(&nh_in), _ros_it(nh_in)
{
}

//
bool RosImagePubSub::add_a_pub(size_t id_in, const std::string& topic_name)
{
  auto result = _image_publisher_map.emplace(id_in, _ros_it.advertise(topic_name, 1));
  return result.second;
}

// output
bool RosImagePubSub::send_image(const int topic_id, const cv::Mat& content_in)
{
  // Content of the message
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", content_in).toImageMsg();
  //
  auto m_it = _image_publisher_map.find(topic_id);
  if (m_it != _image_publisher_map.end())
  {
    _image_publisher_map[topic_id].publish(msg);
    return true;
  }
  else
  {
    return false;
  }
}

bool RosImagePubSub::send_image_rgb(const int topic_id, const cv::Mat& content_in)
{
  cv::Mat mat_img;
  cv::cvtColor(content_in, mat_img, cv::COLOR_RGB2BGR);  //=COLOR_BGRA2RGB

  std_msgs::Header header;  // empty header
  // header.seq = sequence;			 // user defined counter
  header.stamp = ros::Time::now();  // time
  header.frame_id = "camera";       // camera id

  sensor_msgs::Image img_msg;
  cv_bridge::CvImage img_bridge;

  img_bridge = cv_bridge::CvImage(header, "bgr8", mat_img);
  img_bridge.toImageMsg(img_msg);  // from cv_bridge to sensor_msgs::Image

  auto m_it = _image_publisher_map.find(topic_id);
  if (m_it != _image_publisher_map.end())
  {
    _image_publisher_map[topic_id].publish(img_msg);
    return true;
  }
  else
  {
    return false;
  }
}

bool RosImagePubSub::send_image_rgb_gstreamer(const int topic_id, const cv::Mat& content_in)
{
  cv::Mat mat_img;

  mat_img = content_in;  // always is BGR format

  std_msgs::Header header;  // empty header
  // header.seq = sequence;			 // user defined counter
  header.stamp = ros::Time::now();  // time
  header.frame_id = "camera";       // camera id

  sensor_msgs::Image img_msg;
  cv_bridge::CvImage img_bridge;

  img_bridge = cv_bridge::CvImage(header, "bgr8", mat_img);
  img_bridge.toImageMsg(img_msg);  // from cv_bridge to sensor_msgs::Image

  auto m_it = _image_publisher_map.find(topic_id);
  if (m_it != _image_publisher_map.end())
  {
    _image_publisher_map[topic_id].publish(img_msg);
    return true;
  }
  else
  {
    return false;
  }
}
//
bool RosImagePubSub::add_a_sub(size_t id_in, const std::string& topic_name)
{
  auto result = _image_subscriber_map.emplace(
      id_in, _ros_it.subscribe(topic_name, 10, boost::bind(&RosImagePubSub::_CompressedImage_CB, this, _1, id_in),
                               ros::VoidPtr(), image_transport::TransportHints("compressed")));
  if (result.second)
  {
    image_trans_map[id_in];  // This add an default content
  }
  return result.second;
}

//
bool RosImagePubSub::get_image(const int topic_id, cv::Mat& content_out)
{
  if (image_trans_map[topic_id].get(content_out_ptr_map[topic_id]))
  {
    content_out = *(content_out_ptr_map[topic_id]);
    return true;
  }
  else
  {
    return false;
  }
}

void RosImagePubSub::_CompressedImage_CB(const sensor_msgs::ImageConstPtr& msg, const size_t topic_id)
{
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  // put
  image_trans_map[topic_id].put(cv_ptr->image);
}

// // test, image publisher
// std::vector<std::string> image_name_list;
// for (size_t i=0; i < CAMERA_NUM; ++i){
//     image_name_list.emplace_back("image/" + std::to_string(i));
// }
