#include "pedestrian_event.h"

namespace ped
{
void PedestrianEvent::run()
{
  std::thread display_thread(&PedestrianEvent::display_on_terminal, this);
  pedestrian_event();
  display_thread.join();
}

void PedestrianEvent::display_on_terminal()
{
  while (ros::ok() && !PRINT_MESSAGE)
  {
    struct winsize terminal_size;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &terminal_size);
    std::stringstream ss;
    for (int i = 0; i < terminal_size.ws_row; i++)
    {
      ss << "\n";
      if (i == 0 || i == 10 || i == terminal_size.ws_row - 1)
      {
        for (int j = 0; j < terminal_size.ws_col; j++)
        {
          ss << "*";
        }
      }
      else
      {
        std::stringstream line;
        line << "*";
        if (i == 1)
        {
          line << "Time: " << ros::Time::now();
          line << " Delay from camera: " << delay_from_camera_;
        }
        else if (i == 2)
        {
          std::lock_guard<std::mutex> lk(mu_chatter_callback_info_);
          line << chatter_callback_info_;
        }
        else if (i == 3)
        {
          std::lock_guard<std::mutex> lk(mu_front_image_cache_);
          line << "Front camera: buffer size: ";
          line << front_image_cache_.size() << " FPS: ";
          if (!front_image_cache_.empty())
          {
            ros::Time latest_time;
            int frame_number = 0;
            for (int j = front_image_cache_.size() - 1; j >= 0; j--)
            {
              if (ros::Time::now() - front_image_cache_[j].first <= ros::Duration(1))
              {
                latest_time = front_image_cache_[front_image_cache_.size() - 1].first;
                frame_number++;
              }
              else
              {
                break;
              }
            }
            line << frame_number << " time: ";
          }
          else
          {
            line << "NA time: ";
          }

          if (!front_image_cache_.empty())
          {
            line << std::to_string(front_image_cache_[front_image_cache_.size() - 1].first.toSec());
          }
          else
          {
            line << "NA";
          }
        }
        else if (i == 4)
        {
          std::lock_guard<std::mutex> lk(mu_left_image_cache_);
          line << "Left  camera: buffer size: ";
          line << left_image_cache_.size() << " FPS: ";
          if (!left_image_cache_.empty())
          {
            ros::Time latest_time;
            int frame_number = 0;
            for (int j = left_image_cache_.size() - 1; j >= 0; j--)
            {
              if (ros::Time::now() - left_image_cache_[j].first <= ros::Duration(1))
              {
                latest_time = left_image_cache_[left_image_cache_.size() - 1].first;
                frame_number++;
              }
              else
              {
                break;
              }
            }
            line << frame_number << " time: ";
          }
          else
          {
            line << "NA time: ";
          }

          if (!left_image_cache_.empty())
          {
            line << std::to_string(left_image_cache_[left_image_cache_.size() - 1].first.toSec());
          }
          else
          {
            line << "NA";
          }
        }
        else if (i == 5)
        {
          std::lock_guard<std::mutex> lk(mu_right_image_cache_);
          line << "Right camera: buffer size: ";
          line << right_image_cache_.size() << " FPS: ";
          if (!right_image_cache_.empty())
          {
            ros::Time latest_time;
            int frame_number = 0;
            for (int j = right_image_cache_.size() - 1; j >= 0; j--)
            {
              if (ros::Time::now() - right_image_cache_[j].first <= ros::Duration(1))
              {
                latest_time = right_image_cache_[right_image_cache_.size() - 1].first;
                frame_number++;
              }
              else
              {
                break;
              }
            }
            line << frame_number << " time: ";
          }
          else
          {
            line << "NA time: ";
          }

          if (!right_image_cache_.empty())
          {
            line << std::to_string(right_image_cache_[right_image_cache_.size() - 1].first.toSec());
          }
          else
          {
            line << "NA";
          }
        }
        else if (i == 6)
        {
          std::lock_guard<std::mutex> lk(mu_fov30_image_cache_);
          line << "FOV30 camera: buffer size: ";
          line << fov30_image_cache_.size() << " FPS: ";
          if (!fov30_image_cache_.empty())
          {
            ros::Time latest_time;
            int frame_number = 0;
            for (int j = fov30_image_cache_.size() - 1; j >= 0; j--)
            {
              if (ros::Time::now() - fov30_image_cache_[j].first <= ros::Duration(1))
              {
                latest_time = fov30_image_cache_[fov30_image_cache_.size() - 1].first;
                frame_number++;
              }
              else
              {
                break;
              }
            }
            line << frame_number << " time: ";
          }
          else
          {
            line << "NA time: ";
          }

          if (!fov30_image_cache_.empty())
          {
            line << std::to_string(fov30_image_cache_[fov30_image_cache_.size() - 1].first.toSec());
          }
          else
          {
            line << "NA";
          }
        }
        else if (i == 7)
        {
          line << "Planned path size: " << lanelet2_trajectory_.size();
          std::lock_guard<std::mutex> lk(mu_time_nav_path_);
          line << " time: " << time_nav_path_;
        }
        else if (i == 8)
        {
          line << "input_source: " << input_source_ << "   max_distance: " << max_distance_
               << "   show_probability: " << show_probability_;
        }
        else if (i == 9)
        {
          line << "danger_zone_distance: " << danger_zone_distance_ << "   use_2d_for_alarm: " << use_2d_for_alarm_
               << "   skip_frame_number: " << skip_frame_number_;
        }
        else  // i >= 11
        {
          std::lock_guard<std::mutex> lk(mu_ped_info_);
          int size_ped_info_ = ped_info_.size();
          if (i - 9 < size_ped_info_)
          {
            line << ped_info_[i - 9];
          }
        }
        for (int k = line.tellp(); k < terminal_size.ws_col; k++)
        {
          if (k == 0 || k == terminal_size.ws_col - 1)
          {
            line << "*";
          }
          else
          {
            line << " ";
          }
        }
        ss << line.rdbuf();
        line.clear();
        line.str("");
      }
    }
    // std::lock_guard<std::mutex> lock(std::mutex);
    std::cout << ss.rdbuf() << std::flush;
    ss.clear();
    ss.str("");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void PedestrianEvent::veh_info_callback(const msgs::VehInfo::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lk(mu_veh_info_);
#if PRINT_MESSAGE
  ros::Time start;
  start = ros::Time::now();
#endif

  veh_info_ = *msg;

#if PRINT_MESSAGE
  std::cout << "veh_info buffer time cost: " << ros::Time::now() - start << std::endl;
#endif
}

void PedestrianEvent::lanelet2_route_callback(const visualization_msgs::MarkerArray::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lk(mu_lanelet2_route_);
  std::cout << *msg << std::endl;
  ros::Time start;
  start = ros::Time::now();
  lanelet2_route_left_.clear();
  lanelet2_route_right_.clear();
  std::vector<cv::Point3f>().swap(lanelet2_route_left_);
  std::vector<cv::Point3f>().swap(lanelet2_route_right_);
  lanelet2_route_left_.reserve(200);
  lanelet2_route_right_.reserve(200);

  for (auto const& obj : msg->markers)
  {
    if (obj.ns.compare("left_lane_bound") == 0)
    {
      for (auto const& obj_point : obj.points)
      {
        cv::Point3f point;
        point.x = obj_point.x;
        point.y = obj_point.y;
        point.z = obj_point.z;
        bool push_or_not = true;
        for (unsigned int i = 0; i < lanelet2_route_left_.size(); i++)
        {
          if (lanelet2_route_left_[i].x == point.x && lanelet2_route_left_[i].y == point.y &&
              lanelet2_route_left_[i].z == point.z)
          {
            push_or_not = false;
          }
        }
        if (push_or_not)
        {
          lanelet2_route_left_.push_back(point);
        }
      }
    }
    else if (obj.ns.compare("right_lane_bound") == 0)
    {
      for (auto const& obj_point : obj.points)
      {
        cv::Point3f point;
        point.x = obj_point.x;
        point.y = obj_point.y;
        point.z = obj_point.z;
        bool push_or_not = true;
        for (unsigned int i = 0; i < lanelet2_route_right_.size(); i++)
        {
          if (lanelet2_route_right_[i].x == point.x && lanelet2_route_right_[i].y == point.y &&
              lanelet2_route_right_[i].z == point.z)
          {
            push_or_not = false;
          }
        }
        if (push_or_not)
        {
          lanelet2_route_right_.push_back(point);
        }
      }
    }
  }
  std::lock_guard<std::mutex> lk2(mu_time_nav_path_);
  time_nav_path_ = std::to_string(start.toSec());
#if PRINT_MESSAGE
  std::cout << "Path buffer time cost: " << ros::Time::now() - start << std::endl;
#endif
}

void PedestrianEvent::lanelet2_trajectory_callback(const autoware_planning_msgs::Trajectory::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lk(mu_lanelet2_trajectory_);
  ros::Time start;
  start = ros::Time::now();
  lanelet2_trajectory_.clear();
  std::vector<cv::Point2f>().swap(lanelet2_trajectory_);
  lanelet2_trajectory_.reserve(200);

  for (auto const& obj : msg->points)
  {
    cv::Point2f point;
    point.x = obj.pose.position.x;
    point.y = obj.pose.position.y;
    lanelet2_trajectory_.push_back(point);
  }
  std::lock_guard<std::mutex> lk2(mu_time_nav_path_);
  time_nav_path_ = std::to_string(start.toSec());
#if PRINT_MESSAGE
  std::cout << "Path buffer time cost: " << ros::Time::now() - start << std::endl;
#endif
}

void PedestrianEvent::cache_front_image_callback(const sensor_msgs::Image::ConstPtr& msg)
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

  std::lock_guard<std::mutex> lk(mu_front_image_cache_);
  front_image_cache_.push_back({ msg->header.stamp, msg_decode.clone() });
  msg_decode.release();
#if PRINT_MESSAGE
  std::cout << "Image buffer time cost: " << ros::Time::now() - start << std::endl;
  std::cout << "Image buffer size: " << front_image_cache_.size() << std::endl;
#endif
}

void PedestrianEvent::cache_left_image_callback(const sensor_msgs::Image::ConstPtr& msg)
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

  std::lock_guard<std::mutex> lk(mu_left_image_cache_);
  left_image_cache_.push_back({ msg->header.stamp, msg_decode.clone() });
  msg_decode.release();
#if PRINT_MESSAGE
  std::cout << "Image buffer time cost: " << ros::Time::now() - start << std::endl;
  std::cout << "Image buffer size: " << left_image_cache_.size() << std::endl;
#endif
}

void PedestrianEvent::cache_right_image_callback(const sensor_msgs::Image::ConstPtr& msg)
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

  std::lock_guard<std::mutex> lk(mu_right_image_cache_);
  right_image_cache_.push_back({ msg->header.stamp, msg_decode.clone() });
  msg_decode.release();
#if PRINT_MESSAGE
  std::cout << "Image buffer time cost: " << ros::Time::now() - start << std::endl;
  std::cout << "Image buffer size: " << right_image_cache_.size() << std::endl;
#endif
}

void PedestrianEvent::cache_fov30_image_callback(const sensor_msgs::Image::ConstPtr& msg)
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

  std::lock_guard<std::mutex> lk(mu_fov30_image_cache_);
  fov30_image_cache_.push_back({ msg->header.stamp, msg_decode.clone() });
  msg_decode.release();
#if PRINT_MESSAGE
  std::cout << "Image buffer time cost: " << ros::Time::now() - start << std::endl;
  std::cout << "Image buffer size: " << fov30_image_cache_.size() << std::endl;
#endif
}

void PedestrianEvent::front_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  // 0 for front
  main_callback(msg, front_image_cache_, 0, skeleton_buffer_front_);
}
void PedestrianEvent::left_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  // 1 for left
  main_callback(msg, left_image_cache_, 1, skeleton_buffer_left_);
}

void PedestrianEvent::right_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  // 2 for right
  main_callback(msg, right_image_cache_, 2, skeleton_buffer_right_);
}

void PedestrianEvent::fov30_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  // 3 for fov30
  main_callback(msg, fov30_image_cache_, 3, skeleton_buffer_fov30_);
}

void PedestrianEvent::main_callback(const msgs::DetectedObjectArray::ConstPtr& msg,
                                    boost::circular_buffer<std::pair<ros::Time, cv::Mat>>& image_cache, int from_camera,
                                    std::vector<SkeletonBuffer>& skeleton_buffer)
{
  if (!image_cache.empty())  // do if there is image in buffer
  {
    ros::Time inference_start, inference_stop;
    ros::Time start, stop;
    start = ros::Time::now();
    int count_peds = 0;

    // keep original image
    cv::Mat matrix;
    // for painting
    cv::Mat matrix2;
    bool get_timestamp = false;
    ros::Time msgs_timestamp;
    std::vector<msgs::PedObject> ped_objs;
    std::vector<msgs::DetectedObject> alert_objs;
    ped_objs.reserve(msg->objects.end() - msg->objects.begin());

    for (auto const& obj : msg->objects)
    {
      // Only first object need to check raw image
      if (!get_timestamp)
      {
        if (obj.header.stamp.toSec() > 1)
        {
          msgs_timestamp = obj.header.stamp;
        }
        else if (msg->header.stamp.toSec() > 1)
        {
          msgs_timestamp = msg->header.stamp;
        }
        else
        {
          msgs_timestamp = ros::Time::now();
        }

        std::lock_guard<std::mutex> lk(mu_front_image_cache_);
        std::lock_guard<std::mutex> lk2(mu_fov30_image_cache_);
        std::lock_guard<std::mutex> lk3(mu_left_image_cache_);
        std::lock_guard<std::mutex> lk4(mu_right_image_cache_);
        // compare and get the raw image
        for (int i = image_cache.size() - 1; i >= 0; i--)
        {
          if (image_cache[i].first <= msgs_timestamp || i == 0)
          {
#if PRINT_MESSAGE
            std::cout << "GOT CHA !!!!! time: " << image_cache[i].first << " , " << msgs_timestamp << std::endl;
#endif

            matrix = image_cache[i].second;
            // for drawing bbox and keypoints
            matrix.copyTo(matrix2);
            get_timestamp = true;
            break;
          }
        }
      }

      if (obj.classId != 1)
      {  // 1 for people
        continue;
      }

      // set msg infomation
      msgs::PedObject obj_pub;
      obj_pub.header = obj.header;
      obj_pub.header.frame_id = obj.header.frame_id;
      obj_pub.header.stamp = obj.header.stamp;
      obj_pub.classId = obj.classId;
      obj_pub.camInfo = obj.camInfo;
      obj_pub.bPoint = obj.bPoint;
      obj_pub.track.id = obj.track.id;

      if (filter(obj.bPoint, msg->header.stamp))
      {
        obj_pub.crossProbability = -1;
        obj_pub.camInfo.u *= scaling_ratio_width_;
        obj_pub.camInfo.v *= scaling_ratio_height_;
        obj_pub.camInfo.width *= scaling_ratio_width_;
        obj_pub.camInfo.height *= scaling_ratio_height_;
      }
      else
      {
        msgs::DetectedObject alert_obj;
        alert_obj.header = obj.header;
        alert_obj.header.frame_id = obj.header.frame_id;
        alert_obj.header.stamp = obj.header.stamp;
        alert_obj.classId = obj.classId;
        alert_obj.camInfo = obj.camInfo;
        alert_obj.bPoint = obj.bPoint;
        alert_obj.track.id = obj.track.id;
#if PRINT_MESSAGE
        std::cout << "Track ID: " << obj.track.id << std::endl;
#endif

        cv::Mat croped_image;
        // resize from 1920*1208 to 608*384
        obj_pub.camInfo.u *= scaling_ratio_width_;
        obj_pub.camInfo.v *= scaling_ratio_height_;
        obj_pub.camInfo.width *= scaling_ratio_width_;
        obj_pub.camInfo.height *= scaling_ratio_height_;
        // obj_pub.camInfo.v -= 5;
        // obj_pub.camInfo.height += 10;
        // Avoid index out of bounds
        if (obj_pub.camInfo.u + obj_pub.camInfo.width > matrix.cols)
        {
          obj_pub.camInfo.width = matrix.cols - obj_pub.camInfo.u;
        }
        if (obj_pub.camInfo.v + obj_pub.camInfo.height > matrix.rows)
        {
          obj_pub.camInfo.height = matrix.rows - obj_pub.camInfo.v;
        }

        // check bounding box is legal
        if (obj_pub.camInfo.width == 0 || obj_pub.camInfo.height == 0)
        {
          continue;
        }
        count_peds++;
#if PRINT_MESSAGE
        std::cout << matrix.cols << " " << matrix.rows << " " << obj_pub.camInfo.u << " " << obj_pub.camInfo.v << " "
                  << obj_pub.camInfo.u + obj_pub.camInfo.width << " " << obj_pub.camInfo.v + obj_pub.camInfo.height
                  << std::endl;
#endif
        // crop image for openpose
        matrix.copyTo(croped_image);
        croped_image =
            croped_image(cv::Rect(obj_pub.camInfo.u, obj_pub.camInfo.v, obj_pub.camInfo.width, obj_pub.camInfo.height));

        // set size to resize cropped image for openpose
        // max pixel of width or height can only be 368
        int max_pixel = 368;
        float aspect_ratio = 0.0;
        int resize_height_to = 0;
        int resize_width_to = 0;
        if (croped_image.cols >= croped_image.rows)
        {  // width larger than height
          if (croped_image.cols > max_pixel)
          {
            resize_width_to = max_pixel;
          }
          else
          {
            resize_width_to = croped_image.cols;
          }
          resize_width_to = max_pixel;  // force to max pixel
          aspect_ratio = (float)croped_image.rows / (float)croped_image.cols;
          resize_height_to = int(aspect_ratio * resize_width_to);
        }
        else
        {  // height larger than width
          if (croped_image.rows > max_pixel)
          {
            resize_height_to = max_pixel;
          }
          else
          {
            resize_height_to = croped_image.rows;
          }
          resize_height_to = max_pixel;  // force to max pixel
          aspect_ratio = (float)croped_image.cols / (float)croped_image.rows;
          resize_width_to = int(aspect_ratio * resize_height_to);
        }
        cv::resize(croped_image, croped_image, cv::Size(resize_width_to, resize_height_to));
        inference_start = ros::Time::now();
        // search index in skeleton buffer
        int skeleton_index = -1;
        std::vector<cv::Point2f> keypoints;
        {
          std::lock_guard<std::mutex> lk(mu_skeleton_buffer_);

          for (unsigned int i = 0; i < skeleton_buffer.size(); i++)
          {
            if (skeleton_buffer.at(i).track_id == obj_pub.track.id)
            {
              skeleton_index = i;
            }
          }

          if (skeleton_index == -1)  // if there is no data in skeleton buffer
          {
            SkeletonBuffer new_person;
            new_person.timestamp = msg->header.stamp;
            new_person.track_id = obj_pub.track.id;

            keypoints = get_openpose_keypoint(croped_image);

            cv::Point2f zero_keypoint;
            zero_keypoint.x = 0;
            zero_keypoint.y = 0;
            std::vector<cv::Point2f> zero_keypoints;
            zero_keypoints.reserve(number_keypoints_);
            for (unsigned int i = 0; i < number_keypoints_; i++)
            {
              zero_keypoints.emplace_back(zero_keypoint);
            }
            new_person.stored_skeleton.reserve(frame_num_);
            for (unsigned int i = 0; i < frame_num_ - 1; i++)
            {
              new_person.stored_skeleton.emplace_back(zero_keypoints);
            }
            new_person.stored_skeleton.emplace_back(keypoints);

            msgs::PredictSkeleton srv_skip_frame;

            for (unsigned int i = 0; i < new_person.stored_skeleton.size(); i++)
            {
              msgs::Keypoints msgs_keypoints;
              for (unsigned int j = 0; j < new_person.stored_skeleton.at(i).size(); j++)
              {
                msgs::Keypoint msgs_keypoint;
                msgs_keypoint.x = new_person.stored_skeleton.at(i).at(j).x;
                msgs_keypoint.y = new_person.stored_skeleton.at(i).at(j).y;
                msgs_keypoints.keypoint.emplace_back(msgs_keypoint);
              }
              srv_skip_frame.request.original_keypoints.emplace_back(msgs_keypoints);
            }

            // call skip_frame service
            skip_frame_client_.call(srv_skip_frame);

            // get data return from skip_frame service
            for (unsigned int i = 0; i < srv_skip_frame.response.predicted_keypoints.size(); i++)
            {
              std::vector<cv::Point2f> predict_keypoints;
              for (unsigned int j = 0; j < srv_skip_frame.response.predicted_keypoints.at(i).keypoint.size(); j++)
              {
                cv::Point2f predict_keypoint;
                predict_keypoint.x = srv_skip_frame.response.predicted_keypoints.at(i).keypoint.at(j).x;
                predict_keypoint.y = srv_skip_frame.response.predicted_keypoints.at(i).keypoint.at(j).y;
                predict_keypoints.emplace_back(predict_keypoint);
              }
              new_person.calculated_skeleton.emplace_back(predict_keypoints);
              predict_keypoints.clear();
              std::vector<cv::Point2f>().swap(predict_keypoints);
            }

            std::vector<float> bbox;
            bbox.emplace_back(0);
            bbox.emplace_back(0);
            bbox.emplace_back(0);
            bbox.emplace_back(0);
            for (unsigned int i = 0; i < frame_num_ - 1; i++)
            {
              new_person.data_bbox.emplace_back(bbox);
            }
            // last bbox will add after.
            bbox.clear();
            std::vector<float>().swap(bbox);

            skeleton_buffer.emplace_back(new_person);
            skeleton_index = skeleton_buffer.size() - 1;
          }
          else  // if there is data in skeleton buffer
          {
            // if there is data in skeleton buffer but calculated_skeleton is already empty
            if (skeleton_buffer.at(skeleton_index).calculated_skeleton.empty())
            {
              if (!skeleton_buffer.at(skeleton_index).stored_skeleton.empty())
              {
                skeleton_buffer.at(skeleton_index)
                    .stored_skeleton.erase(skeleton_buffer.at(skeleton_index).stored_skeleton.begin());
              }

              keypoints = get_openpose_keypoint(croped_image);

              skeleton_buffer.at(skeleton_index).stored_skeleton.emplace_back(keypoints);

              msgs::PredictSkeleton srv_skip_frame;

              for (unsigned int i = 0; i < skeleton_buffer.at(skeleton_index).stored_skeleton.size(); i++)
              {
                msgs::Keypoints msgs_keypoints;
                for (unsigned int j = 0; j < skeleton_buffer.at(skeleton_index).stored_skeleton.at(i).size(); j++)
                {
                  msgs::Keypoint msgs_keypoint;
                  msgs_keypoint.x = skeleton_buffer.at(skeleton_index).stored_skeleton.at(i).at(j).x;
                  msgs_keypoint.y = skeleton_buffer.at(skeleton_index).stored_skeleton.at(i).at(j).y;
                  msgs_keypoints.keypoint.emplace_back(msgs_keypoint);
                }
                srv_skip_frame.request.original_keypoints.emplace_back(msgs_keypoints);
              }

              // call skip_frame service
              skip_frame_client_.call(srv_skip_frame);

              skeleton_buffer.at(skeleton_index).calculated_skeleton.clear();
              std::vector<std::vector<cv::Point2f>>().swap(skeleton_buffer.at(skeleton_index).calculated_skeleton);

              // get predicted_keypoints return from skip_frame service
              for (unsigned int i = 0; i < srv_skip_frame.response.predicted_keypoints.size(); i++)
              {
                std::vector<cv::Point2f> predict_keypoints;
                for (unsigned int j = 0; j < srv_skip_frame.response.predicted_keypoints.at(i).keypoint.size(); j++)
                {
                  cv::Point2f predict_keypoint;
                  predict_keypoint.x = srv_skip_frame.response.predicted_keypoints.at(i).keypoint.at(j).x;
                  predict_keypoint.y = srv_skip_frame.response.predicted_keypoints.at(i).keypoint.at(j).y;
                  predict_keypoints.emplace_back(predict_keypoint);
                }
                skeleton_buffer.at(skeleton_index).calculated_skeleton.emplace_back(predict_keypoints);
                predict_keypoints.clear();
                std::vector<cv::Point2f>().swap(predict_keypoints);
              }

              // replace original stored_skeleton only when service return size is correct
              if (srv_skip_frame.response.processed_keypoints.size() == frame_num_)
              {
                skeleton_buffer.at(skeleton_index).stored_skeleton.clear();
                std::vector<std::vector<cv::Point2f>>().swap(skeleton_buffer.at(skeleton_index).stored_skeleton);
                // get processed_keypoints return from skip_frame service
                for (unsigned int i = 0; i < srv_skip_frame.response.processed_keypoints.size(); i++)
                {
                  skeleton_buffer.at(skeleton_index).timestamp = msg->header.stamp;
                  std::vector<cv::Point2f> back_predict_keypoints;
                  for (unsigned int j = 0; j < srv_skip_frame.response.processed_keypoints.at(i).keypoint.size(); j++)
                  {
                    cv::Point2f back_predict_keypoint;
                    back_predict_keypoint.x = srv_skip_frame.response.processed_keypoints.at(i).keypoint.at(j).x;
                    back_predict_keypoint.y = srv_skip_frame.response.processed_keypoints.at(i).keypoint.at(j).y;
                    back_predict_keypoints.emplace_back(back_predict_keypoint);
                  }
                  skeleton_buffer.at(skeleton_index).stored_skeleton.emplace_back(back_predict_keypoints);
                  back_predict_keypoints.clear();
                  std::vector<cv::Point2f>().swap(back_predict_keypoints);
                }
              }
            }
            else  // there is data in skeleton_buffer and still has calculated_skeleton
            {
              if (!skeleton_buffer.at(skeleton_index).stored_skeleton.empty())
              {
                skeleton_buffer.at(skeleton_index)
                    .stored_skeleton.erase(skeleton_buffer.at(skeleton_index).stored_skeleton.begin());
              }
              keypoints = skeleton_buffer.at(skeleton_index).calculated_skeleton.at(0);
              skeleton_buffer.at(skeleton_index)
                  .calculated_skeleton.erase(skeleton_buffer.at(skeleton_index).calculated_skeleton.begin());

              // to fit camInfo u, v, width, height
              double w_h_ratio = (double)obj_pub.camInfo.width / (double)obj_pub.camInfo.height;
              double min_x = 0;
              double min_y = 0;
              double max_x = 0;
              double max_y = 1;
              for (unsigned int i = 0; i < keypoints.size(); i++)
              {
                min_x = std::min(min_x, (double)keypoints.at(i).x);
                min_y = std::min(min_y, (double)keypoints.at(i).y);
                max_x = std::max(max_x, (double)keypoints.at(i).x);
                max_y = std::max(max_y, (double)keypoints.at(i).y);
              }
              double max_w = max_x - min_x;
              double max_h = max_y - min_y;
              for (unsigned int i = 0; i < keypoints.size(); i++)
              {
                if (keypoints.at(i).x != 0 && keypoints.at(i).y != 0)
                {
                  keypoints.at(i).x = (keypoints.at(i).x - min_x) / max_w * w_h_ratio;
                  keypoints.at(i).y = (keypoints.at(i).y - min_y) / max_h;
                }
              }
              skeleton_buffer.at(skeleton_index).stored_skeleton.emplace_back(keypoints);
            }
          }
          std::vector<float> bbox;
          bbox.emplace_back(obj.camInfo.u);
          bbox.emplace_back(obj.camInfo.v);
          bbox.emplace_back(obj.camInfo.u + obj.camInfo.width);
          bbox.emplace_back(obj.camInfo.v + obj.camInfo.height);
          skeleton_buffer.at(skeleton_index).data_bbox.emplace_back(bbox);
          if (skeleton_buffer.at(skeleton_index).data_bbox.size() > frame_num_)
          {
            skeleton_buffer.at(skeleton_index).data_bbox.erase(skeleton_buffer.at(skeleton_index).data_bbox.begin());
          }

          inference_stop = ros::Time::now();

          bool has_keypoint = false;
          int count_points = 0;
          int body_part[13] = { 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14 };
          unsigned int body_part_size = sizeof(body_part) / sizeof(*body_part);
          for (unsigned int i = 0; i < body_part_size; i++)
          {
            if (keypoints.at(body_part[i]).x != 0 || keypoints.at(body_part[i]).y != 0)
            {
              count_points++;
              if (count_points >= 3)
              {
                has_keypoint = true;
              }
            }
          }
          if (has_keypoint)
          {
            msgs::PredictCrossing srv_pedcorss_tf;
            // prepare bboxes for ros service
            for (unsigned int i = 0; i < skeleton_buffer.at(skeleton_index).data_bbox.size(); i++)
            {
              msgs::CamInfo msgs_bbox;
              msgs_bbox.u = skeleton_buffer.at(skeleton_index).data_bbox.at(i).at(0);
              msgs_bbox.v = skeleton_buffer.at(skeleton_index).data_bbox.at(i).at(1);
              msgs_bbox.width = skeleton_buffer.at(skeleton_index).data_bbox.at(i).at(2) - msgs_bbox.u;
              msgs_bbox.height = skeleton_buffer.at(skeleton_index).data_bbox.at(i).at(3) - msgs_bbox.v;
              srv_pedcorss_tf.request.bboxes.emplace_back(msgs_bbox);
            }
            // prepare keypoints for ros service
            for (unsigned int i = 0; i < skeleton_buffer.at(skeleton_index).stored_skeleton.size(); i++)
            {
              msgs::Keypoints msgs_keypoints;
              for (unsigned int j = 0; j < skeleton_buffer.at(skeleton_index).stored_skeleton.at(i).size(); j++)
              {
                msgs::Keypoint msgs_keypoint;
                msgs_keypoint.x = skeleton_buffer.at(skeleton_index).stored_skeleton.at(i).at(j).x;
                msgs_keypoint.y = skeleton_buffer.at(skeleton_index).stored_skeleton.at(i).at(j).y;
                msgs_keypoints.keypoint.emplace_back(msgs_keypoint);
              }
              srv_pedcorss_tf.request.keypoints.emplace_back(msgs_keypoints);
            }
            // automatically switch from LSTM and Random Forest
            if (tf_client_.call(srv_pedcorss_tf))
            {
              obj_pub.crossProbability = srv_pedcorss_tf.response.result_0;
              std::cout << "predict (LSTM): " << obj_pub.crossProbability << srv_pedcorss_tf.response.result_0 << std::endl;
            }
            else
            {
              obj_pub.crossProbability = crossing_predict(skeleton_buffer.at(skeleton_index).data_bbox,
                                                        skeleton_buffer.at(skeleton_index).stored_skeleton);
              std::cout << "predict (RF): " << obj_pub.crossProbability << std::endl;
            }
          }
          clean_old_skeleton_buffer(skeleton_buffer, msg->header.stamp);
        }

        obj_pub.facing_direction = get_facing_direction(keypoints);
        // obj_pub.body_direction = get_body_direction(keypoints);

        // only for front camera
        if (from_camera == 0)
        {
          obj_pub.crossProbability = adjust_probability(obj_pub);
        }

        // copy another nav_path to prevent vector changing while calculating
        if (obj_pub.bPoint.p0.x != 0 || obj_pub.bPoint.p0.y != 0)
        {
          msgs::PointXYZ camera_position = obj_pub.bPoint.p0;
          std::vector<cv::Point2f> nav_path_temp;
          {
            std::lock_guard<std::mutex> lk(mu_lanelet2_trajectory_);
            nav_path_temp.assign(lanelet2_trajectory_.begin(), lanelet2_trajectory_.end());
          }
          geometry_msgs::TransformStamped transform_stamped;
          try
          {
            transform_stamped = tf_buffer_.lookupTransform("map", "base_link", msg->header.stamp, ros::Duration(0.5));
#if PRINT_MESSAGE
            std::cout << transform_stamped << std::endl;
#endif
          }
          catch (tf2::TransformException& ex)
          {
            ROS_WARN("%s", ex.what());
            ros::Duration(1.0).sleep();
            return;
          }
          geometry_msgs::PoseStamped point_in;
          point_in.pose.position.x = camera_position.x;
          point_in.pose.position.y = camera_position.y;
          point_in.pose.position.z = camera_position.z;
          geometry_msgs::PoseStamped point_out;
          tf2::doTransform(point_in, point_out, transform_stamped);
          camera_position.x = point_out.pose.position.x;
          camera_position.y = point_out.pose.position.y;
          camera_position.z = point_out.pose.position.z;
          // find the nearest nav_path point from pedestian's position
          cv::Point2f nearest_point;
          double min_distance_from_path = 100000;
          for (const cv::Point2f& path_point : nav_path_temp)
          {
            // calculate distance between pedestrian and each nav_path point
            double distance_diff = get_distance2(path_point.x, path_point.y, camera_position.x, camera_position.y);
            if (min_distance_from_path > distance_diff)
            {
              min_distance_from_path = distance_diff;
              nearest_point.x = path_point.x;
              nearest_point.y = path_point.y;
            }
          }
          // store distance from plan path
          /*
          {
            std::lock_guard<std::mutex> lk(mu_skeleton_buffer);

            for (unsigned int i = 0; i < skeleton_buffer.size(); i++)
            {
              if (skeleton_buffer.at(i).track_id == obj_pub.track.id)
              {
                skeleton_buffer.at(i).history_distance_from_path.emplace_back(min_distance_from_path);
                if (skeleton_buffer.at(i).history_distance_from_path.size() > 10)
                {
                  skeleton_buffer.at(i).history_distance_from_path.erase(
                      skeleton_buffer.at(i).history_distance_from_path.begin());
                  if (skeleton_buffer.at(i).history_distance_from_path.at(
                          skeleton_buffer.at(i).history_distance_from_path.size() - 1) -
                          skeleton_buffer.at(i).history_distance_from_path.at(0) <
                      -0.5)
                  {
                    obj_pub.crossProbability += 10;
                  }
                  if (skeleton_buffer.at(i).history_distance_from_path.at(
                          skeleton_buffer.at(i).history_distance_from_path.size() - 1) -
                          skeleton_buffer.at(i).history_distance_from_path.at(0) >
                      0.5)
                  {
                    obj_pub.crossProbability = -10;
                  }
                }
                break;
              }
            }
          }
          */

          // too close to planned path
          // from center to left and right 2 meters
          if (min_distance_from_path < danger_zone_distance_)
          {
            obj_pub.crossProbability = 1;
          }
          if (obj_pub.crossProbability * 100 >= cross_threshold_)
          {
            double distance_from_car = 0;
            cv::Point2f previous_path_point;
            bool passed_car_head = false;
            for (const cv::Point2f& path_point : nav_path_temp)
            {
              // check
              if (path_point.x > 0)
              {
                passed_car_head = true;
              }
              // add distance between points
              if (passed_car_head)
              {
                distance_from_car +=
                    get_distance2(path_point.x, path_point.y, previous_path_point.x, previous_path_point.y);
              }
              if (path_point.x == nearest_point.x && path_point.y == nearest_point.y)
              {
                std::lock_guard<std::mutex> lk(mu_veh_info_);
#if DUMP_LOG
                // print distance
                // file_ << ros::Time::now() << "," << obj_pub.track.id << "," << distance_from_car << ","
                //  << veh_info_.ego_speed << "\n";
#endif
#if PRINT_MESSAGE
                std::cout << "same, distance: " << distance_from_car << " id: " << obj_pub.track.id
                          << " time: " << ros::Time::now() << " speed: " << veh_info_.ego_speed << std::endl;
#endif
                break;
              }
              previous_path_point = path_point;
            }
            // to free memory from vector
            nav_path_temp.erase(nav_path_temp.begin(), nav_path_temp.end());
            std::vector<cv::Point2f>().swap(nav_path_temp);

            geometry_msgs::TransformStamped transform_stamped;
            try
            {
              transform_stamped = tf_buffer_.lookupTransform("base_link", "map", msg->header.stamp, ros::Duration(0.5));
#if PRINT_MESSAGE
              std::cout << transform_stamped << std::endl;
#endif
            }
            catch (tf2::TransformException& ex)
            {
              ROS_WARN("%s", ex.what());
              ros::Duration(1.0).sleep();
              return;
            }
            geometry_msgs::PoseStamped point_in;
            point_in.pose.position.x = nearest_point.x;
            point_in.pose.position.y = nearest_point.y;
            point_in.pose.position.z = 0;
            geometry_msgs::PoseStamped point_out;
            tf2::doTransform(point_in, point_out, transform_stamped);
            nearest_point.x = point_out.pose.position.x;
            nearest_point.y = point_out.pose.position.y;

            double diff_x = (nearest_point.x - obj_pub.bPoint.p0.x) / 10;
            double diff_y = (nearest_point.y - obj_pub.bPoint.p0.y) / 10;
            alert_obj.track.forecasts.reserve(20);
            obj_pub.track.forecasts.reserve(20);
            alert_obj.track.is_ready_prediction = 1;
            obj_pub.track.is_ready_prediction = 1;
            for (int i = 0; i < 20; i++)
            {
              msgs::PathPrediction pp;
              pp.position.x = obj_pub.bPoint.p0.x + diff_x * i;
              pp.position.y = obj_pub.bPoint.p0.y + diff_y * i;
#if PRINT_MESSAGE
              std::cout << pp.position << std::endl;
#endif
              alert_obj.track.forecasts.push_back(pp);
              obj_pub.track.forecasts.push_back(pp);
            }
            alert_objs.push_back(alert_obj);
          }
        }

        for (auto point : keypoints)
        {
          msgs::Keypoint kp;
          kp.x = point.x;
          kp.y = point.y;
          obj_pub.keypoints.emplace_back(kp);
        }
        keypoints.clear();
        std::vector<cv::Point2f>().swap(keypoints);
      }
      ped_objs.emplace_back(obj_pub);
    }

    msgs::DetectedObjectArray alert_obj_array;

    alert_obj_array.header = msg->header;
    alert_obj_array.header.frame_id = msg->header.frame_id;
    alert_obj_array.header.stamp = msgs_timestamp;
    alert_obj_array.objects.assign(alert_objs.begin(), alert_objs.end());
    if (from_camera == 0)  // front
    {
      alert_pub_front_.publish(alert_obj_array);
    }
    else if (from_camera == 1)  // left
    {
      alert_pub_left_.publish(alert_obj_array);
    }
    else if (from_camera == 2)  // right
    {
      alert_pub_right_.publish(alert_obj_array);
    }
    else if (from_camera == 3)  // fov30
    {
      alert_pub_fov30_.publish(alert_obj_array);
    }

    msgs::PedObjectArray ped_obj_array;
    // ped_obj_array.raw_image = img_msg;
    ped_obj_array.header = msg->header;
    ped_obj_array.header.frame_id = msg->header.frame_id;
    if (msgs_timestamp.toSec() > 1)
    {
      ped_obj_array.header.stamp = msgs_timestamp;
    }
    else
    {
      ped_obj_array.header.stamp = msg->header.stamp;
    }
    
    ped_obj_array.objects.assign(ped_objs.begin(), ped_objs.end());
    if (from_camera == 0)  // front
    {
      chatter_pub_front_.publish(ped_obj_array);
    }
    else if (from_camera == 1)  // left
    {
      chatter_pub_left_.publish(ped_obj_array);
    }
    else if (from_camera == 2)  // right
    {
      chatter_pub_right_.publish(ped_obj_array);
    }
    else if (from_camera == 3)  // fov30
    {
      chatter_pub_fov30_.publish(ped_obj_array);
    }

    alert_objs.clear();
    std::vector<msgs::DetectedObject>().swap(alert_objs);
    ped_objs.clear();
    std::vector<msgs::PedObject>().swap(ped_objs);
    matrix.release();
    matrix2.release();

    stop = ros::Time::now();

    std::lock_guard<std::mutex> lk(mu_chatter_callback_info_);
    average_inference_time_ = average_inference_time_ * 0.9 + (inference_stop - inference_start).toSec() * 0.1;
    count_++;
    chatter_callback_info_ = "Cost time: " + std::to_string((stop - start).toSec()) +
                             "(sec) OpenPose inference time: " + std::to_string(average_inference_time_) +
                             "(sec) Loop: " + std::to_string(count_);

    std::lock_guard<std::mutex> lk2(mu_delay_from_camera_);
    total_time_ += stop - start;
    delay_from_camera_ = std::to_string((ros::Time::now() - msgs_timestamp).toSec());
#if DUMP_LOG
    // print inference time
    file_ << ros::Time::now() << "," << count_peds << "," << (inference_stop - inference_start).toSec() << ","
          << delay_from_camera_ << "\n";
#endif
#if PRINT_MESSAGE
    std::cout << "Delay from camera: " << delay_from_camera_ << std::endl;
    std::cout << "Camera source: " << from_camera << std::endl;
    std::cout << chatter_callback_info_ << std::endl;
    std::cout << "Total time: " << total_time_ << " sec / loop: " << count_ << std::endl;
#endif
    // sleep 0.1 sec for 10 FPS
    /*
    ros::Duration sleep_time = ros::Duration(0.1) - (stop - start);
    if (sleep_time > ros::Duration(0.0))
    {
      sleep_time.sleep();
    }
    */
  }
}

float PedestrianEvent::adjust_probability(msgs::PedObject obj)
{
  // pedestrian position
  float x = obj.camInfo.u + obj.camInfo.width * 0.5;
  float y = obj.camInfo.v + obj.camInfo.height;
  // too far away from car
  if (y < 275)
  {
    return obj.crossProbability * 0.7;
  }
  if (obj.facing_direction == 4)
  {
    return obj.crossProbability;
  }
  // pedestrian in danger zone, force determine as Cross
  if (std::fabs(x - 303) < 100 * (y - 275) / 108)
  {
    if (use_2d_for_alarm_)
    {
      if (y >= 310)
      {
        return 1;
      }
      else
      {
        // in danger zone but too far away from car
        return obj.crossProbability;
      }
    }
    else
    {
      return obj.crossProbability;
    }
  }
  // at right sidewalk but not walking to left
  if (x >= 303 && std::fabs(x - 303) >= 300 * (y - 275) / 108 && obj.facing_direction != 0)
  {
    return 0;
  }
  // at left sidewalk but not walking to right
  if (x < 303 && std::fabs(x - 303) >= 550 * (y - 275) / 108 && obj.facing_direction != 1)
  {
#if PRINT_MESSAGE
    std::cout << "at left sidewalk but not walking to right" << std::endl;
#endif
    return 0;
  }
  // walking into danger zone
  // at right half of car and not goint right hand side
  if (x >= 303 && obj.facing_direction != 1)
  {
    return obj.crossProbability;
  }
  // at left half of car and not goint left hand side
  if (x < 303 && obj.facing_direction != 0)
  {
    return obj.crossProbability;
  }
  // pedestrian walking out from danger zone
  if (std::fabs(x - 303) < 200 * (y - 275) / 108)
  {
    return obj.crossProbability * 0.8;
  }

  return obj.crossProbability * 0.7;
}

void PedestrianEvent::draw_ped_front_callback(const msgs::PedObjectArray::ConstPtr& msg)
{
  draw_pedestrians_callback(msg, front_image_cache_, 0);
}

void PedestrianEvent::draw_ped_left_callback(const msgs::PedObjectArray::ConstPtr& msg)
{
  draw_pedestrians_callback(msg, left_image_cache_, 1);
}

void PedestrianEvent::draw_ped_right_callback(const msgs::PedObjectArray::ConstPtr& msg)
{
  draw_pedestrians_callback(msg, right_image_cache_, 2);
}

void PedestrianEvent::draw_ped_fov30_callback(const msgs::PedObjectArray::ConstPtr& msg)
{
  draw_pedestrians_callback(msg, fov30_image_cache_, 3);
}

void PedestrianEvent::draw_pedestrians_callback(const msgs::PedObjectArray::ConstPtr& msg,
                                                boost::circular_buffer<std::pair<ros::Time, cv::Mat>>& image_cache,
                                                int from_camera)
{
  if (image_cache.empty())  // do if there is image in buffer
  {
    return;
  }

  cv::Mat matrix;
  cv::Mat matrix2;
  ros::Time msgs_timestamp = ros::Time::now();
  if (!msg->objects.empty())
  {
    if (msg->objects[0].header.stamp.toSec() > 1)
    {
      msgs_timestamp = msg->objects[0].header.stamp;
    }
    else
    {
      msgs_timestamp = msg->header.stamp;
    }
  }
  else if (msg->header.stamp.toSec() > 1)
  {
    msgs_timestamp = msg->header.stamp;
  }
  {
    std::lock_guard<std::mutex> lk(mu_front_image_cache_);
    std::lock_guard<std::mutex> lk2(mu_fov30_image_cache_);
    std::lock_guard<std::mutex> lk3(mu_left_image_cache_);
    std::lock_guard<std::mutex> lk4(mu_right_image_cache_);
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
  }

  for (const auto& obj : msg->objects)
  {
    cv::Rect box;
    box.x = obj.camInfo.u;
    box.y = obj.camInfo.v;
    box.width = obj.camInfo.width;
    box.height = obj.camInfo.height;
    if (obj.crossProbability >= 0)
    {
      cv::rectangle(matrix, box.tl(), box.br(), CV_RGB(0, 255, 0), 2);
    }
    else
    {
      cv::rectangle(matrix, box.tl(), box.br(), CV_RGB(100, 100, 100), 2);
    }
    std::vector<cv::Point2f> keypoints;
    int keypoint_number = 0;
    for (auto const& point : obj.keypoints)
    {
      cv::Point2f kp;
      kp.x = point.x;
      kp.y = point.y;
      keypoints.emplace_back(kp);
    }

    if (keypoints.size() < 18)
    {
      continue;
    }

    // draw keypoints on raw image
    int body_part[17] = { 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
    unsigned int body_part_size = sizeof(body_part) / sizeof(*body_part);
    for (unsigned int i = 0; i < body_part_size; i++)
    {
      keypoints.at(body_part[i]).x = keypoints.at(body_part[i]).x * obj.camInfo.height;
      keypoints.at(body_part[i]).y = keypoints.at(body_part[i]).y * obj.camInfo.height;
      if (keypoint_is_detected(keypoints.at(body_part[i])))
      {
        cv::Point p = keypoints.at(body_part[i]);
        p.x = obj.camInfo.u + p.x;
        p.y = obj.camInfo.v + p.y;
        cv::circle(matrix, p, 2, cv::Scalar(0, 255, 0), -1);
        keypoint_number++;
      }
    }
    // draw hands
    int body_part1[7] = { 4, 3, 2, 1, 5, 6, 7 };
    unsigned int body_part1_size = sizeof(body_part1) / sizeof(*body_part1);
    for (unsigned int i = 0; i < body_part1_size - 1; i++)
    {
      if (keypoint_is_detected(keypoints.at(body_part1[i])) && keypoint_is_detected(keypoints.at(body_part1[i + 1])))
      {
        cv::Point p = keypoints.at(body_part1[i]);
        p.x = obj.camInfo.u + p.x;
        p.y = obj.camInfo.v + p.y;
        cv::Point p2 = keypoints.at(body_part1[i + 1]);
        p2.x = obj.camInfo.u + p2.x;
        p2.y = obj.camInfo.v + p2.y;
        cv::line(matrix, p, p2, cv::Scalar(0, 0, 255), 1);
      }
    }
    // draw legs
    int body_part2[7] = { 11, 10, 9, 1, 12, 13, 14 };
    unsigned int body_part2_size = sizeof(body_part2) / sizeof(*body_part2);
    for (unsigned int i = 0; i < body_part2_size - 1; i++)
    {
      if (keypoint_is_detected(keypoints.at(body_part2[i])) && keypoint_is_detected(keypoints.at(body_part2[i + 1])))
      {
        cv::Point p = keypoints.at(body_part2[i]);
        p.x = obj.camInfo.u + p.x;
        p.y = obj.camInfo.v + p.y;
        cv::Point p2 = keypoints.at(body_part2[i + 1]);
        p2.x = obj.camInfo.u + p2.x;
        p2.y = obj.camInfo.v + p2.y;
        cv::line(matrix, p, p2, cv::Scalar(255, 0, 255), 1);
      }
    }
    keypoints.clear();
    std::vector<cv::Point2f>().swap(keypoints);
    if (box.y >= 5)
    {
      box.y -= 5;
    }
    else
    {
      box.y = 0;
    }

    std::string probability;
    int p = 100 * obj.crossProbability;
    if (p >= cross_threshold_)
    {
      if (show_probability_)
      {
        probability = "C(" + std::to_string(p / 100) + "." + std::to_string(p / 10 % 10) + std::to_string(p % 10) + ")";
      }
      else
      {
        probability = "C";
      }

      cv::putText(matrix, probability, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1 /*font size*/, cv::Scalar(0, 50, 255), 2,
                  4, false);
    }
    else
    {
      if (show_probability_)
      {
        if (p >= 10)
        {
          probability = "NC(" + std::to_string(p / 100) + "." + std::to_string(p % 100) + ")";
        }
        else
        {
          probability = "NC(" + std::to_string(p / 100) + ".0" + std::to_string(p % 100) + ")";
        }
      }
      else
      {
        probability = "NC";
      }

      cv::putText(matrix, probability, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1 /*font size*/, cv::Scalar(100, 220, 0), 2,
                  4, false);
    }

    if (box.y >= 25)
    {
      box.y -= 25;
    }
    else
    {
      box.y = 0;
    }

    std::string id_print = "[" + std::to_string(obj.track.id % 1000) + "]";

    // draw face direction
    if (obj.facing_direction == 4)  // no direction
    {
      id_print += "     ";
    }
    else if (obj.facing_direction == 0)
    {
      id_print += "left ";
    }
    else if (obj.facing_direction == 1)
    {
      id_print += "right";
    }
    else if (obj.facing_direction == 2)
    {
      id_print += "back ";
    }
    else
    {
      id_print += "front";
    }
    cv::putText(matrix, id_print, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5 /*font size*/, cv::Scalar(100, 220, 0), 1, 2,
                false);

    std::lock_guard<std::mutex> lk(mu_ped_info_);
    ped_info_.insert(ped_info_.begin(), id_print + " " + probability + " x: " + std::to_string((int)obj.bPoint.p0.x) +
                                            " y: " + std::to_string((int)obj.bPoint.p0.y) +
                                            " keypoints number: " + std::to_string(keypoint_number));
    if (ped_info_.size() > 40)
    {
      ped_info_.erase(ped_info_.end() - 1);
    }
  }
  // do resize only when computer cannot support
  // cv::resize(matrix, matrix, cv::Size(matrix.cols / 1, matrix.rows / 1));

  // make cv::Mat to sensor_msgs::Image
  sensor_msgs::ImageConstPtr viz_pub = cv_bridge::CvImage(std_msgs::Header(), "bgr8", matrix).toImageMsg();

  if (from_camera == 0)  // front
  {
    box_pub_front_.publish(viz_pub);
  }
  else if (from_camera == 1)  // left
  {
    box_pub_left_.publish(viz_pub);
  }
  else if (from_camera == 2)  // right
  {
    box_pub_right_.publish(viz_pub);
  }
  else if (from_camera == 3)  // fov30
  {
    box_pub_fov30_.publish(viz_pub);
  }

  matrix.release();
  matrix2.release();
}

/**
 * return
 * 0 for facing left
 * 1 for facing right
 * 2 for facing car side
 * 3 for facing car opposite side
 * 4 for no direction
 */
int PedestrianEvent::get_facing_direction(const std::vector<cv::Point2f>& keypoints)
{
  bool left_ear = keypoint_is_detected(keypoints.at(18));
  bool left_eye = keypoint_is_detected(keypoints.at(16));
  bool right_ear = keypoint_is_detected(keypoints.at(17));
  bool right_eye = keypoint_is_detected(keypoints.at(15));
  bool face_detection[4] = { right_ear, right_eye, left_eye, left_ear };
  for (int i = 0; i < 16; i++)
  {
    if (face_detection[0] == direction_table_[i][0] && face_detection[1] == direction_table_[i][1] &&
        face_detection[2] == direction_table_[i][2] && face_detection[3] == direction_table_[i][3])
    {
      return direction_table_[i][4];
    }
  }
  return 4;
}

/**
 * return
 * 0 for facing left
 * 1 for facing right
 * 2 for facing car side
 * 3 for facing car opposite side
 */
int PedestrianEvent::get_body_direction(const std::vector<cv::Point2f>& keypoints)
{
  int result = 0;
  if (keypoint_is_detected(keypoints.at(9)) && keypoint_is_detected(keypoints.at(10)) &&
      keypoint_is_detected(keypoints.at(11)))
  {
    if ((keypoints.at(9).x + keypoints.at(11).x) / 2 - keypoints.at(10).x > 0)
    {
      result = 0;
    }
    else if (keypoints.at(10).x - (keypoints.at(9).x + keypoints.at(11).x) / 2 > 0)
    {
      result = 1;
    }
    else
    {
      result = 2;
    }
  }
  else
  {
    result = 3;
  }

  if (keypoint_is_detected(keypoints.at(12)) && keypoint_is_detected(keypoints.at(13)) &&
      keypoint_is_detected(keypoints.at(14)))
  {
    if ((keypoints.at(12).x + keypoints.at(14).x) / 2 - keypoints.at(13).x > 0)
    {
      result += 0;
    }
    else if (keypoints.at(13).x - (keypoints.at(12).x + keypoints.at(14).x) / 2 > 0)
    {
      result += 10;
    }
    else
    {
      result += 20;
    }
  }
  else
  {
    result += 30;
  }

  return result;
}

bool PedestrianEvent::keypoint_is_detected(cv::Point2f keypoint)
{
  if (keypoint.x > 0 || keypoint.y > 0)
  {
    return true;
  }
  return false;
}

/**
 * extract features and pass to random forest model
 * return
 * cross probability
 */
float PedestrianEvent::crossing_predict(std::vector<std::vector<float>>& bbox_array,
                                        std::vector<std::vector<cv::Point2f>>& keypoint_array)
{
  try
  {
    // initialize feature
    std::vector<float> feature;

    for (unsigned int index = 0; index < frame_num_; index++)
    {
      // Add bbox to feature vector
      std::vector<float> bbox = bbox_array.at(index);
      feature.insert(feature.end(), bbox.begin(), bbox.end());

      std::vector<cv::Point2f> keypoint = keypoint_array.at(index);
      if (!keypoint.empty())
      {
        std::vector<float> keypoints_x;
        std::vector<float> keypoints_y;

        // Get body keypoints we need
        int body_part[13] = { 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14 };
        int body_part_size = sizeof(body_part) / sizeof(*body_part);
        for (int i = 0; i < body_part_size; i++)
        {
          keypoints_x.insert(keypoints_x.end(), keypoint[body_part[i]].x);
          keypoints_y.insert(keypoints_y.end(), keypoint[body_part[i]].y);
        }

        // Calculate x_distance, y_distance, distance, angle
        for (int m = 0; m < body_part_size; m++)
        {
          for (int n = m + 1; n < body_part_size; n++)
          {
            float dist_x, dist_y, dist, angle;
            if (keypoints_x[m] != 0.0f && keypoints_y[m] != 0.0f && keypoints_x[n] != 0.0f && keypoints_y[n] != 0.0f)
            {
              dist_x = std::fabs(keypoints_x[m] - keypoints_x[n]);
              dist_y = std::fabs(keypoints_y[m] - keypoints_y[n]);
              dist = get_distance2(keypoints_x[m], keypoints_y[m], keypoints_x[n], keypoints_y[n]);
              angle = get_angle2(keypoints_x[m], keypoints_y[m], keypoints_x[n], keypoints_y[n]);
            }
            else
            {
              dist_x = 0.0f;
              dist_y = 0.0f;
              dist = 0.0f;
              angle = 0.0f;
            }
            float input[] = { dist_x, dist_y, dist, angle };
            feature.insert(feature.end(), input, input + sizeof(input) / sizeof(input[0]));
          }
        }

        // Calculate 3 inner angles of each 3 keypoints
        for (int m = 0; m < body_part_size; m++)
        {
          for (int n = m + 1; n < body_part_size; n++)
          {
            for (int k = n + 1; k < body_part_size; k++)
            {
              float angle[3] = { 0.0f, 0.0f, 0.0f };
              float* angle_ptr;
              if ((keypoints_x[m] != 0.0f || keypoints_y[m] != 0.0f) &&
                  (keypoints_x[n] != 0.0f || keypoints_y[n] != 0.0f) &&
                  (keypoints_x[k] != 0.0f || keypoints_y[k] != 0.0f))
              {
                angle_ptr = get_triangle_angle(keypoints_x[m], keypoints_y[m], keypoints_x[n], keypoints_y[n],
                                               keypoints_x[k], keypoints_y[k]);
                angle[0] = *angle_ptr;
                angle[1] = *(angle_ptr + 1);
                angle[2] = *(angle_ptr + 2);
              }
              feature.insert(feature.end(), angle, angle + sizeof(angle) / sizeof(angle[0]));
            }
          }
        }
        keypoints_x.clear();
        std::vector<float>().swap(keypoints_x);
        keypoints_y.clear();
        std::vector<float>().swap(keypoints_y);
      }
      else  // if keypoint is empty
      {
        float* zero_arr;
        // The first four feature are bb_x1, bb_y1, bb_x2, bb_y2
        int other_feature = feature_num_ - 4;
        zero_arr = new float[other_feature]();
        feature.insert(feature.begin(), zero_arr, zero_arr + sizeof(zero_arr) / sizeof(zero_arr[0]));
        delete[] zero_arr;
      }
    }

    // Convert vector to array
    int total_feature_size = feature_num_ * frame_num_;
    float feature_arr[total_feature_size];
    std::copy(feature.begin(), feature.end(), feature_arr);
    // Convert array to Mat
    cv::Mat feature_mat = cv::Mat(1, total_feature_size, CV_32F, feature_arr);
    // Predict
    float predict_result = predict_rf_pose(feature_mat);
    feature_mat.release();
    feature.clear();
    std::vector<float>().swap(feature);
    return predict_result;
  }
  catch (const std::exception& e)
  {
    std::cout << "predict error" << std::endl;
    return 0;
  }
}

// return Euclidian distance between two points
float PedestrianEvent::get_distance2(float x1, float y1, float x2, float y2)
{
  return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

// return degree with line formed by two points and vertical line
float PedestrianEvent::get_angle2(float x1, float y1, float x2, float y2)
{
  return M_PI / 2 - std::atan2(std::fabs(y1 - y2), std::fabs(x1 - x2));
}

// return 3 inner angles of the triangle formed by three points
float* PedestrianEvent::get_triangle_angle(float x1, float y1, float x2, float y2, float x3, float y3)
{
  float a = get_distance2(x1, y1, x2, y2);
  float b = get_distance2(x2, y2, x3, y3);
  float c = get_distance2(x1, y1, x3, y3);
  float test = (a * a + c * c - b * b) / (2 * a * c);
  static float angle[3] = { 0.0f, 0.0f, 0.0f };
  if (test <= 1 && test >= -1)
  {
    angle[0] = std::acos((a * a + c * c - b * b) / (2 * a * c));
    angle[1] = std::acos((a * a + b * b - c * c) / (2 * a * b));
    angle[2] = M_PI - angle[0] - angle[1];
  }
  else
  {
    if (std::max(a, std::max(b, c)) == a)
    {
      angle[2] = M_PI;
    }
    else if (std::max(a, std::max(b, c)) == b)
    {
      angle[0] = M_PI;
    }
    else
    {
      angle[1] = M_PI;
    }
  }
  return angle;
}

// use random forest model to predict cross probability
// return cross probability
float PedestrianEvent::predict_rf_pose(const cv::Mat& input_data)
{
  cv::Mat votes;
  rf_pose_->getVotes(input_data, votes, 0);
  float positive = votes.at<int>(1, 1);
  float negative = votes.at<int>(1, 0);
  float p = positive / (negative + positive);

#if PRINT_MESSAGE
  std::cout << "prediction: " << p << votes.size() << std::endl;
  std::cout << votes.at<int>(0, 0) << " " << votes.at<int>(0, 1) << std::endl;
  std::cout << votes.at<int>(1, 0) << " " << votes.at<int>(1, 1) << std::endl;
#endif
  votes.release();
  return p;
}

bool PedestrianEvent::filter(const msgs::BoxPoint box_point, ros::Time time_stamp)
{
  cv::Point2f position;
  position.x = box_point.p0.x;
  position.y = box_point.p0.y;

  if (position.x > max_distance_ || position.x < -3)
  {
    return true;
  }

  geometry_msgs::TransformStamped transform_stamped;
  try
  {
    transform_stamped = tf_buffer_.lookupTransform("map", "base_link", time_stamp, ros::Duration(0.5));
#if PRINT_MESSAGE
    std::cout << transform_stamped << std::endl;
#endif
  }
  catch (tf2::TransformException& ex)
  {
    ROS_WARN("%s", ex.what());
    ros::Duration(1.0).sleep();
    return false;
  }

  std::vector<cv::Point3f> lanelet2_route_left_temp;
  std::vector<cv::Point3f> lanelet2_route_right_temp;
  {
    std::lock_guard<std::mutex> lk(mu_lanelet2_route_);
    if (lanelet2_route_left_.empty() || lanelet2_route_right_.empty())
    {
      return false;
    }
    else
    {
      lanelet2_route_left_temp.assign(lanelet2_route_left_.begin(), lanelet2_route_left_.end());
      lanelet2_route_right_temp.assign(lanelet2_route_right_.begin(), lanelet2_route_right_.end());
    }
  }
  std::vector<cv::Point2f> route_left_transformed;
  std::vector<cv::Point2f> route_right_transformed;
  for (auto const& obj : lanelet2_route_left_temp)
  {
    cv::Point2f point;
    point.x = obj.x;
    point.y = obj.y;
    route_left_transformed.push_back(point);
  }
  for (auto const& obj : lanelet2_route_right_temp)
  {
    cv::Point2f point;
    point.x = obj.x;
    point.y = obj.y;
    route_right_transformed.push_back(point);
  }
  lanelet2_route_left_temp.clear();
  lanelet2_route_right_temp.clear();
  std::vector<cv::Point3f>().swap(lanelet2_route_left_temp);
  std::vector<cv::Point3f>().swap(lanelet2_route_right_temp);
  geometry_msgs::PoseStamped point_in;
  point_in.pose.position.x = position.x;
  point_in.pose.position.y = position.y;
  point_in.pose.position.z = 0;
  geometry_msgs::PoseStamped point_out;
  tf2::doTransform(point_in, point_out, transform_stamped);
  position.x = point_out.pose.position.x;
  position.y = point_out.pose.position.y;
  // expand warning zone for left bound
  double expand_range_left = 5;
  std::vector<cv::Point2f> expanded_route_left;
  for (unsigned int i = 0; i < route_left_transformed.size(); i++)
  {
    if (i == 0)
    {
      double diff_x;
      double diff_y;
      diff_x = route_left_transformed[0].x - route_right_transformed[0].x;
      diff_y = route_left_transformed[0].y - route_right_transformed[0].y;
      double distance = sqrt(pow(diff_x, 2) + pow(diff_y, 2));
      cv::Point2f expand_point = route_left_transformed[i];
      expand_point.x = expand_point.x + diff_x / distance * expand_range_left;
      expand_point.y = expand_point.y + diff_y / distance * expand_range_left;
      expanded_route_left.push_back(expand_point);
    }
    else if (i == route_left_transformed.size() - 1)
    {
      double diff_x;
      double diff_y;
      diff_x = route_left_transformed.back().x - route_right_transformed.back().x;
      diff_y = route_left_transformed.back().y - route_right_transformed.back().y;
      double distance = sqrt(pow(diff_x, 2) + pow(diff_y, 2));
      cv::Point2f expand_point = route_left_transformed[i];
      expand_point.x = expand_point.x + diff_x / distance * expand_range_left;
      expand_point.y = expand_point.y + diff_y / distance * expand_range_left;
      expanded_route_left.push_back(expand_point);
    }
    else
    {
      double diff_x;
      double diff_y;
      diff_x = route_left_transformed[i + 1].x - route_left_transformed[i - 1].x;
      diff_y = route_left_transformed[i + 1].y - route_left_transformed[i - 1].y;
      double N_x = (-1) * diff_y;
      double N_y = diff_x;
      double distance = sqrt(pow(N_x, 2) + pow(N_y, 2));
      cv::Point2f expand_point = route_left_transformed[i];
      expand_point.x = expand_point.x + N_x / distance * expand_range_left;
      expand_point.y = expand_point.y + N_y / distance * expand_range_left;
      expanded_route_left.push_back(expand_point);
    }
  }
  // expand warning zone for right bound
  double expand_range_right = 2;
  std::vector<cv::Point2f> expanded_route_right;
  for (unsigned int i = 0; i < route_right_transformed.size(); i++)
  {
    if (i == 0)
    {
      double diff_x;
      double diff_y;
      diff_x = route_right_transformed[0].x - route_left_transformed[0].x;
      diff_y = route_right_transformed[0].y - route_left_transformed[0].y;
      double distance = sqrt(pow(diff_x, 2) + pow(diff_y, 2));
      cv::Point2f expand_point = route_right_transformed[i];
      expand_point.x = expand_point.x + diff_x / distance * expand_range_right;
      expand_point.y = expand_point.y + diff_y / distance * expand_range_right;
      expanded_route_right.push_back(expand_point);
    }
    else if (i == route_left_transformed.size() - 1)
    {
      double diff_x;
      double diff_y;
      diff_x = route_right_transformed.back().x - route_left_transformed.back().x;
      diff_y = route_right_transformed.back().y - route_left_transformed.back().y;
      double distance = sqrt(pow(diff_x, 2) + pow(diff_y, 2));
      cv::Point2f expand_point = route_right_transformed[i];
      expand_point.x = expand_point.x + diff_x / distance * expand_range_right;
      expand_point.y = expand_point.y + diff_y / distance * expand_range_right;
      expanded_route_right.push_back(expand_point);
    }
    else
    {
      double diff_x;
      double diff_y;
      diff_x = route_right_transformed[i + 1].x - route_right_transformed[i - 1].x;
      diff_y = route_right_transformed[i + 1].y - route_right_transformed[i - 1].y;
      double N_x = diff_y;
      double N_y = (-1) * diff_x;
      double distance = sqrt(pow(N_x, 2) + pow(N_y, 2));
      cv::Point2f expand_point = route_right_transformed[i];
      expand_point.x = expand_point.x + N_x / distance * expand_range_right;
      expand_point.y = expand_point.y + N_y / distance * expand_range_right;
      expanded_route_right.push_back(expand_point);
    }
  }
  route_left_transformed.clear();
  std::vector<cv::Point2f>().swap(route_left_transformed);
  route_right_transformed.clear();
  std::vector<cv::Point2f>().swap(route_right_transformed);
  // route_right_transformed add into route_left_transformed reversed
  while (!expanded_route_right.empty())
  {
    expanded_route_left.push_back(expanded_route_right.back());
    expanded_route_right.pop_back();
  }
  expanded_route_right.clear();
  std::vector<cv::Point2f>().swap(expanded_route_right);
  expanded_route_left.push_back(expanded_route_left[0]);  // close the polygon

  geometry_msgs::PolygonStamped polygon_merker;
  polygon_merker.header.frame_id = "map";
  for (auto const& obj : expanded_route_left)
  {
    geometry_msgs::Point32 polygon_point;
    polygon_point.x = obj.x;
    polygon_point.y = obj.y;
    polygon_merker.polygon.points.push_back(polygon_point);
  }
  warning_zone_pub_.publish(polygon_merker);
  // all route, check ped in polygon or not
  // no need to filter peds in warning zone
  if (check_in_polygon(position, expanded_route_left))
  {
    expanded_route_left.clear();
    std::vector<cv::Point2f>().swap(expanded_route_left);
    return true;
  }
  else
  {
    expanded_route_left.clear();
    std::vector<cv::Point2f>().swap(expanded_route_left);
    return false;
  }
  expanded_route_left.clear();
  std::vector<cv::Point2f>().swap(expanded_route_left);
  // no need to filter
  return false;
}

bool PedestrianEvent::check_in_polygon(cv::Point2f position, std::vector<cv::Point2f>& polygon)
{
  int nvert = polygon.size();
  double testx = position.x;
  double testy = position.y;
  std::vector<double> vertx;
  std::vector<double> verty;
  for (auto const& obj : polygon)
  {
    vertx.emplace_back(obj.x);
    verty.emplace_back(obj.y);
  }

  int i, j, c = 0;
  for (i = 0, j = nvert - 1; i < nvert; j = i++)
  {
    if (((verty[i] > testy) != (verty[j] > testy)) &&
        (testx < (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i]))
    {
      c = 1 + c;
      ;
    }
  }
  vertx.clear();
  std::vector<double>().swap(vertx);
  verty.clear();
  std::vector<double>().swap(verty);
  if (c % 2 == 0)
  {
    return true;
  }
  else
  {
    return false;
  }
}

void PedestrianEvent::clean_old_skeleton_buffer(std::vector<SkeletonBuffer>& skeleton_buffer, ros::Time msg_timestamp)
{
  for (unsigned int i = 0; i < skeleton_buffer.size(); i++)
  {
    if (msg_timestamp - skeleton_buffer.at(i).timestamp > ros::Duration(10))
    {
      skeleton_buffer.erase(skeleton_buffer.begin() + i);
      i--;
    }
  }
}

void PedestrianEvent::pedestrian_event()
{
  // AsyncSpinner reference:
  //  https://gist.github.com/bgromov/45ebeced9e8067d9f13cceececc00d5b#file-test_spinner-cpp-L63

  // custom callback queue
  ros::CallbackQueue queue_1;
  ros::CallbackQueue queue_2;
  ros::CallbackQueue queue_3;
  ros::CallbackQueue queue_4;
  ros::CallbackQueue queue_5;
  ros::CallbackQueue queue_6;
  ros::CallbackQueue queue_7;
  ros::CallbackQueue queue_8;
  ros::CallbackQueue queue_9;
  ros::CallbackQueue queue_10;
  ros::CallbackQueue queue_11;
  ros::CallbackQueue queue_12;
  ros::CallbackQueue queue_13;
  // This node handle uses global callback queue
  ros::NodeHandle nh_sub_1;
  // and this one uses custom queue
  ros::NodeHandle nh_sub_2;
  ros::NodeHandle nh_sub_3;
  ros::NodeHandle nh_sub_4;
  ros::NodeHandle nh_sub_5;
  ros::NodeHandle nh_sub_6;
  ros::NodeHandle nh_sub_7;
  ros::NodeHandle nh_sub_8;
  ros::NodeHandle nh_sub_9;
  ros::NodeHandle nh_sub_10;
  ros::NodeHandle nh_sub_11;
  ros::NodeHandle nh_sub_12;
  ros::NodeHandle nh_sub_13;
  ros::NodeHandle nh_sub_14;
  // Set custom callback queue
  nh_sub_2.setCallbackQueue(&queue_1);
  nh_sub_3.setCallbackQueue(&queue_2);
  nh_sub_4.setCallbackQueue(&queue_3);
  nh_sub_5.setCallbackQueue(&queue_4);
  nh_sub_6.setCallbackQueue(&queue_5);
  nh_sub_7.setCallbackQueue(&queue_6);
  nh_sub_8.setCallbackQueue(&queue_7);
  nh_sub_9.setCallbackQueue(&queue_8);
  nh_sub_10.setCallbackQueue(&queue_9);
  nh_sub_11.setCallbackQueue(&queue_10);
  nh_sub_12.setCallbackQueue(&queue_11);
  nh_sub_13.setCallbackQueue(&queue_12);
  nh_sub_14.setCallbackQueue(&queue_13);

  ros::Subscriber sub_1;
  ros::Subscriber sub_2;
  ros::Subscriber sub_3;
  ros::Subscriber sub_4;
  ros::Subscriber sub_5;
  ros::Subscriber sub_6;
  ros::Subscriber sub_7;
  ros::Subscriber sub_8;
  ros::Subscriber sub_9;
  ros::Subscriber sub_10;
  ros::Subscriber sub_11;
  ros::Subscriber sub_12;
  ros::Subscriber sub_13;
  ros::Subscriber sub_14;
  if (input_source_ == 4)  // if (input_source_ == 4)
  {
    sub_1 = nh_sub_1.subscribe("/Tracking2D/front_bottom_60", 1, &PedestrianEvent::front_callback,
                               this);  // /Tracking2D/front_bottom_60 is subscirbe topic
    sub_2 = nh_sub_2.subscribe("/Tracking2D/left_back_60", 1, &PedestrianEvent::left_callback,
                               this);  // /Tracking2D/left_back_60 is subscirbe topic
    sub_3 = nh_sub_3.subscribe("/Tracking2D/right_back_60", 1, &PedestrianEvent::right_callback,
                               this);  // /Tracking2D/right_back_60 is subscirbe topic
    sub_4 = nh_sub_4.subscribe("/Tracking2D/front_top_far_30", 1, &PedestrianEvent::fov30_callback,
                               this);  // /Tracking2D/right_back_60 is subscirbe topic
    sub_5 = nh_sub_5.subscribe("/cam/front_bottom_60", 1, &PedestrianEvent::cache_front_image_callback,
                               this);  // /cam/F_right is subscirbe topic
    sub_6 = nh_sub_6.subscribe("/cam/left_back_60", 1, &PedestrianEvent::cache_left_image_callback,
                               this);  // /cam/F_center is subscirbe topic
    sub_7 = nh_sub_7.subscribe("/cam/right_back_60", 1, &PedestrianEvent::cache_right_image_callback,
                               this);  // /cam/F_center is subscirbe topic
    sub_8 =
        nh_sub_8.subscribe("/planning/scenario_planning/trajectory", 1, &PedestrianEvent::lanelet2_trajectory_callback,
                           this);  // /cam/F_center is subscirbe topic
    sub_9 = nh_sub_9.subscribe("/planning/mission_planning/route_marker", 1, &PedestrianEvent::lanelet2_route_callback,
                               this);  // /cam/F_center is subscirbe topic
    sub_10 = nh_sub_10.subscribe("/PedCross/Pedestrians/front_bottom_60", 1, &PedestrianEvent::draw_ped_front_callback,
                                 this);  // /cam/F_center is subscirbe topic
    sub_11 = nh_sub_11.subscribe("/PedCross/Pedestrians/left_back_60", 1, &PedestrianEvent::draw_ped_left_callback,
                                 this);  // /cam/F_center is subscirbe topic
    sub_12 = nh_sub_12.subscribe("/PedCross/Pedestrians/right_back_60", 1, &PedestrianEvent::draw_ped_right_callback,
                                 this);  // /cam/F_center is subscirbe topic
    sub_13 = nh_sub_13.subscribe("/cam/front_top_far_30", 1, &PedestrianEvent::cache_fov30_image_callback,
                                 this);  // /cam/F_center is subscirbe topic
    sub_14 = nh_sub_14.subscribe("/PedCross/Pedestrians/front_top_far_30", 1, &PedestrianEvent::draw_ped_fov30_callback,
                                 this);  // /cam/F_center is subscirbe topic
  }

  // Create AsyncSpinner, run it on all available cores and make it process custom callback queue
  async_spinner_1_.reset(new ros::AsyncSpinner(0, &queue_1));
  async_spinner_2_.reset(new ros::AsyncSpinner(0, &queue_2));
  async_spinner_3_.reset(new ros::AsyncSpinner(0, &queue_3));
  async_spinner_4_.reset(new ros::AsyncSpinner(0, &queue_4));
  async_spinner_5_.reset(new ros::AsyncSpinner(0, &queue_5));
  async_spinner_6_.reset(new ros::AsyncSpinner(0, &queue_6));
  async_spinner_7_.reset(new ros::AsyncSpinner(0, &queue_7));
  async_spinner_8_.reset(new ros::AsyncSpinner(0, &queue_8));
  async_spinner_9_.reset(new ros::AsyncSpinner(0, &queue_9));
  async_spinner_10_.reset(new ros::AsyncSpinner(0, &queue_10));
  async_spinner_11_.reset(new ros::AsyncSpinner(0, &queue_11));
  async_spinner_12_.reset(new ros::AsyncSpinner(0, &queue_12));
  async_spinner_13_.reset(new ros::AsyncSpinner(0, &queue_13));

  spinner_trigger_ = true;

  // Loop with 100 Hz rate
  ros::Rate loop_rate(20);
  while (ros::ok())
  {
    // Enable state changed
    if (spinner_trigger_)
    {
      // Clear old callback from the queue
      queue_1.clear();
      queue_2.clear();
      queue_3.clear();
      queue_4.clear();
      queue_5.clear();
      queue_6.clear();
      queue_7.clear();
      queue_8.clear();
      queue_9.clear();
      queue_10.clear();
      queue_11.clear();
      queue_12.clear();
      queue_13.clear();
      // Start the spinner
      async_spinner_1_->start();
      async_spinner_2_->start();
      async_spinner_3_->start();
      async_spinner_4_->start();
      async_spinner_5_->start();
      async_spinner_6_->start();
      async_spinner_7_->start();
      async_spinner_8_->start();
      async_spinner_9_->start();
      async_spinner_10_->start();
      async_spinner_11_->start();
      async_spinner_12_->start();
      async_spinner_13_->start();
      ROS_INFO("Spinner enabled");
      // Reset trigger
      spinner_trigger_ = false;
    }

    // Process messages on global callback queue
    ros::spinOnce();
    loop_rate.sleep();
  }
  // Release AsyncSpinner object
  async_spinner_1_.reset();
  async_spinner_2_.reset();
  async_spinner_3_.reset();
  async_spinner_4_.reset();
  async_spinner_5_.reset();
  async_spinner_6_.reset();
  async_spinner_7_.reset();
  async_spinner_8_.reset();
  async_spinner_9_.reset();
  async_spinner_10_.reset();
  async_spinner_11_.reset();
  async_spinner_12_.reset();
  async_spinner_13_.reset();
  // Wait for ROS threads to terminate
  ros::waitForShutdown();
}

// return 25 keypoints detected by openpose
std::vector<cv::Point2f> PedestrianEvent::get_openpose_keypoint(cv::Mat& input_image)
{
#if PRINT_MESSAGE
  ros::Time timer = ros::Time::now();
#endif

  std::vector<cv::Point2f> points;
  points.reserve(number_keypoints_);

  float height = input_image.rows;

  std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datum_to_process = create_datum(input_image);
  bool successfully_emplaced = openpose_.waitAndEmplace(datum_to_process);
  std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datum_processed;
  if (successfully_emplaced && openpose_.waitAndPop(datum_processed))
  {
    // display(datum_processed);
    if (datum_processed != nullptr && !datum_processed->empty())
    {
#if PRINT_MESSAGE
      op::opLog("\nKeypoints:");
#endif
      // Accesing each element of the keypoints
      const auto& pose_keypoints = datum_processed->at(0)->poseKeypoints;
#if PRINT_MESSAGE
      op::opLog("Person pose keypoints:");
#endif
      for (auto person = 0; person < 1; person++)  // only get first detected person
      {
#if PRINT_MESSAGE
        op::opLog("Person " + std::to_string(person) + " (x, y, score):");
#endif
        for (auto body_part = 0; body_part < pose_keypoints.getSize(1); body_part++)
        {
          float x = pose_keypoints[{ person, body_part, 0 }] / height;
          float y = pose_keypoints[{ person, body_part, 1 }] / height;
          points.emplace_back(cv::Point2f(x, y));

          std::string value_to_print;
          for (auto xyscore = 0; xyscore < pose_keypoints.getSize(2); xyscore++)
          {
            value_to_print += std::to_string(pose_keypoints[{ person, body_part, xyscore }]) + " ";
          }
#if PRINT_MESSAGE
          op::opLog(value_to_print);
#endif
        }
      }
    }
  }

#if PRINT_MESSAGE
  std::cout << "Openpose time cost: " << ros::Time::now() - timer << std::endl;
#endif

  for (int i = points.size(); i < 25; i++)
  {
    points.emplace_back(cv::Point2f(0.0, 0.0));
  }

  return points;
}

bool PedestrianEvent::display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datums_ptr)
{
  // User's displaying/saving/other processing here
  // datum.cvOutputData: rendered frame with pose or heatmaps
  // datum.pose_keypoints: Array<float> with the estimated pose
  char key = ' ';
  if (datums_ptr != nullptr && !datums_ptr->empty())
  {
    cv::imshow("User worker GUI", OP_OP2CVCONSTMAT(datums_ptr->at(0)->cvOutputData));
    // Display image and sleeps at least 1 ms (it usually sleeps ~5-10 msec to display the image)
    key = (char)cv::waitKey(1);
  }
  else
  {
    op::opLog("Nullptr or empty datums_ptr found.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
  }
  return (key == 27);
}

std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> PedestrianEvent::create_datum(cv::Mat& mat)
{
  // Create new datum
  auto datums_ptr = std::make_shared<std::vector<std::shared_ptr<op::Datum>>>();
  datums_ptr->emplace_back();
  auto& datum_ptr = datums_ptr->at(0);
  datum_ptr = std::make_shared<op::Datum>();

  // Fill datum
  datum_ptr->cvInputData = OP_CV2OPCONSTMAT(mat);

  return datums_ptr;
}

int PedestrianEvent::openposeROS()
{
  // logging_level
  op::checkBool(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__,
                __FUNCTION__, __FILE__);
  op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
  op::Profiler::setDefaultX(FLAGS_profile_speed);

  op::opLog("Starting pose estimation thread(s)", op::Priority::High);
  openpose_.start();

  return 0;
}
}  // namespace ped
int main(int argc, char** argv)
{
#if USE_GLOG
  google::InstallFailureSignalHandler();
#endif
  ros::Time::init();
  ros::Time start, stop;
  start = ros::Time::now();
  ros::init(argc, argv, "pedestrian_event");

  ped::PedestrianEvent pe;
  tf2_ros::TransformListener tf_listener(pe.tf_buffer_);
  std::cout << PED_MODEL_DIR + std::string("/rf_10frames_normalization_15peek.yml") << std::endl;
  pe.rf_pose_ = cv::ml::StatModel::load<cv::ml::RTrees>(PED_MODEL_DIR + std::string("/rf_10frames_normalization_15peek."
                                                                                    "yml"));

  ros::NodeHandle nh1;
  pe.chatter_pub_front_ = nh1.advertise<msgs::PedObjectArray>("/PedCross/Pedestrians/front_bottom_60",
                                                              1);  // /PedCross/Pedestrians is pub topic
  ros::NodeHandle nh2;
  pe.box_pub_front_ =
      nh2.advertise<sensor_msgs::Image&>("/PedCross/DrawBBox/front_bottom_60", 1);  // /PedCross/DrawBBox is pub topic
  ros::NodeHandle nh3;
  pe.chatter_pub_left_ = nh3.advertise<msgs::PedObjectArray>("/PedCross/Pedestrians/left_back_60",
                                                             1);  // /PedCross/Pedestrians is pub topic
  ros::NodeHandle nh4;
  pe.box_pub_left_ =
      nh4.advertise<sensor_msgs::Image&>("/PedCross/DrawBBox/left_back_60", 1);  // /PedCross/DrawBBox is pub topic
  ros::NodeHandle nh5;
  pe.chatter_pub_right_ = nh5.advertise<msgs::PedObjectArray>("/PedCross/Pedestrians/right_back_60",
                                                              1);  // /PedCross/Pedestrians is pub topic
  ros::NodeHandle nh6;
  pe.box_pub_right_ =
      nh6.advertise<sensor_msgs::Image&>("/PedCross/DrawBBox/right_back_60", 1);  // /PedCross/DrawBBox is pub topic
  ros::NodeHandle nh7;
  pe.alert_pub_front_ = nh7.advertise<msgs::DetectedObjectArray>("/PedCross/Alert/front_bottom_60",
                                                                 1);  // /PedCross/DrawBBox is pub topic
  pe.alert_pub_left_ =
      nh7.advertise<msgs::DetectedObjectArray>("/PedCross/Alert/left_back_60", 1);  // /PedCross/DrawBBox is pub topic
  pe.alert_pub_right_ =
      nh7.advertise<msgs::DetectedObjectArray>("/PedCross/Alert/right_back_60", 1);  // /PedCross/DrawBBox is pub topic
  pe.alert_pub_fov30_ = nh7.advertise<msgs::DetectedObjectArray>("/PedCross/Alert/front_top_far_30",
                                                                 1);  // /PedCross/DrawBBox is pub topic
  ros::NodeHandle nh8;
  pe.warning_zone_pub_ =
      nh8.advertise<geometry_msgs::PolygonStamped>("/PedCross/Polygon", 1);  // /PedCross/DrawBBox is pub topic
  ros::NodeHandle nh9;
  pe.chatter_pub_fov30_ = nh9.advertise<msgs::PedObjectArray>("/PedCross/Pedestrians/front_top_far_30",
                                                              1);  // /PedCross/Pedestrians is pub topic
  ros::NodeHandle nh10;
  pe.box_pub_fov30_ =
      nh10.advertise<sensor_msgs::Image&>("/PedCross/DrawBBox/front_top_far_30", 1);  // /PedCross/DrawBBox is pub topic
  // Get parameters from ROS
  ros::NodeHandle nh;
  nh.param<bool>("/pedestrian_event/show_probability", pe.show_probability_, true);
  nh.param<int>("/pedestrian_event/input_source", pe.input_source_, 4);
  nh.param<double>("/pedestrian_event/max_distance", pe.max_distance_, 50);
  nh.param<double>("/pedestrian_event/danger_zone_distance", pe.danger_zone_distance_, 2);
  nh.param<bool>("/pedestrian_event/use_2d_for_alarm", pe.use_2d_for_alarm_, false);
  nh.param<int>("/pedestrian_event/crossing_threshold", pe.cross_threshold_, 55);
  nh.param<int>("/skip_frame_server/skip_frame_number", pe.skip_frame_number_, 1);

  pe.skip_frame_client_ = nh.serviceClient<msgs::PredictSkeleton>("skip_frame");
  pe.tf_client_ = nh.serviceClient<msgs::PredictCrossing>("pedcross_tf");

  pe.front_image_cache_ = boost::circular_buffer<std::pair<ros::Time, cv::Mat>>(pe.buffer_size_);
  pe.left_image_cache_ = boost::circular_buffer<std::pair<ros::Time, cv::Mat>>(pe.buffer_size_);
  pe.right_image_cache_ = boost::circular_buffer<std::pair<ros::Time, cv::Mat>>(pe.buffer_size_);
  pe.fov30_image_cache_ = boost::circular_buffer<std::pair<ros::Time, cv::Mat>>(pe.buffer_size_);

  pe.openposeROS();

  stop = ros::Time::now();
  std::cout << "PedCross started. Init time: " << stop - start << " sec" << std::endl;
  pe.count_ = 0;

#if DUMP_LOG
  std::stringstream ss;
  ss << "../../../ped_output.csv";
  std::string fname = ss.str();
  pe.file_.open(fname, std::ios_base::app);
#endif

  pe.run();

#if DUMP_LOG
  pe.file_.close();
#endif

  return 0;
}
