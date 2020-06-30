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
  while (ros::ok())
  {
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &terminal_size);
    std::stringstream ss;
    for (int i = 0; i < terminal_size.ws_row; i++)
    {
      ss << "\n";
      if (i == 0 || i == 8 || i == terminal_size.ws_row - 1)
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
          line << " Delay from camera: " << delay_from_camera;
        }
        else if (i == 2)
        {
          line << chatter_callback_info;
        }
        else if (i == 3)
        {
          line << "Image buffer size: ";
          line << front_image_cache.size() << "(608x384) ";
          line << crop_image_cache.size() << "(1920x314) time: ";
          if (front_image_cache.size() > 0)
          {
            line << std::to_string(front_image_cache[front_image_cache.size() - 1].first.toSec());
          }
          else
          {
            line << "NA";
          }
        }
        else if (i == 4)
        {
          line << "Image FPS: ";
          if (front_image_cache.size() > 0)
          {
            ros::Time latest_time;
            int frame_number = 0;
            for (int j = front_image_cache.size() - 1; j >= 0; j--)
            {
              if (ros::Time::now() - front_image_cache[j].first <= ros::Duration(1))
              {
                latest_time = front_image_cache[front_image_cache.size() - 1].first;
                frame_number++;
              }
              else
              {
                break;
              }
            }
            line << frame_number << "(608x384) ";
          }
          else
          {
            line << "NA(608x384) ";
          }
          if (crop_image_cache.size() > 0)
          {
            ros::Time latest_time;
            int frame_number = 0;
            for (int j = crop_image_cache.size() - 1; j >= 0; j--)
            {
              if (ros::Time::now() - crop_image_cache[j].first <= ros::Duration(1))
              {
                latest_time = crop_image_cache[crop_image_cache.size() - 1].first;
                frame_number++;
              }
              else
              {
                break;
              }
            }
            line << frame_number << "(1920x314)";
          }
          else
          {
            line << "NA(1920x314)";
          }
        }
        else if (i == 5)
        {
          line << "Planned path size: " << nav_path.size();
        }
        else if (i == 6)
        {
          line << "input_source: " << input_source << "   max_distance: " << max_distance << "   show_probability: " << show_probability;
        }
        else if (i == 7)
        {
          line << "danger_zone_distance: " << danger_zone_distance << "   use_2d_for_alarm: " << use_2d_for_alarm;
        }
        else
        {
          int size_ped_info = ped_info.size();
          if (i - 9 < size_ped_info)
          {
            line << ped_info[i - 9];
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
    std::lock_guard<std::mutex> lock(std::mutex);
    std::cout << ss.rdbuf() << std::flush;
    ss.clear();
    ss.str("");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void PedestrianEvent::veh_info_callback(const msgs::VehInfo::ConstPtr& msg)
{
#if PRINT_MESSAGE
  ros::Time start;
  start = ros::Time::now();
#endif

  veh_info = *msg;

#if PRINT_MESSAGE
  std::cout << "veh_info buffer time cost: " << ros::Time::now() - start << std::endl;
#endif
}

void PedestrianEvent::nav_path_callback(const nav_msgs::Path::ConstPtr& msg)
{
#if PRINT_MESSAGE
  ros::Time start;
  start = ros::Time::now();
#endif
  nav_path.clear();
  std::vector<geometry_msgs::PoseStamped>().swap(nav_path);
  nav_path.reserve(200);
  for (auto const& obj : msg->poses)
  {
    geometry_msgs::PoseStamped point = obj;
    nav_path.push_back(point);
  }
#if PRINT_MESSAGE
  std::cout << "Path buffer time cost: " << ros::Time::now() - start << std::endl;
#endif
}

void PedestrianEvent::cache_crop_image_callback(const sensor_msgs::Image::ConstPtr& msg)
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
  crop_image_cache.push_back({ msg->header.stamp, msg_decode });

#if PRINT_MESSAGE
  std::cout << "Crop Image buffer time cost: " << ros::Time::now() - start << std::endl;
#endif
}

void PedestrianEvent::cache_front_image_callback(const sensor_msgs::Image::ConstPtr& msg)
{
  cache_image_callback(msg, front_image_cache);
}

void PedestrianEvent::cache_left_image_callback(const sensor_msgs::Image::ConstPtr& msg)
{
  cache_image_callback(msg, left_image_cache);
}

void PedestrianEvent::cache_right_image_callback(const sensor_msgs::Image::ConstPtr& msg)
{
  cache_image_callback(msg, right_image_cache);
}

void PedestrianEvent::cache_image_callback(const sensor_msgs::Image::ConstPtr& msg, boost::circular_buffer<std::pair<ros::Time, cv::Mat>> &image_cache)
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
  image_cache.push_back({ msg->header.stamp, msg_decode });

#if PRINT_MESSAGE
  std::cout << "Image buffer time cost: " << ros::Time::now() - start << std::endl;
#endif
}

void PedestrianEvent::front_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  // 0 for front
  main_callback(msg, buffer_front, front_image_cache, 0);
}
void PedestrianEvent::left_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  // 1 for left
  main_callback(msg, buffer_left, left_image_cache, 1);
}

void PedestrianEvent::right_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  // 2 for right
  main_callback(msg, buffer_right, right_image_cache, 2);
}

void PedestrianEvent::main_callback(const msgs::DetectedObjectArray::ConstPtr& msg, Buffer &buffer, boost::circular_buffer<std::pair<ros::Time, cv::Mat>> &image_cache, int from_camera)
{
  if (!image_cache.empty() && !nav_path.empty())  // do if there is image in buffer
  {
    count++;
    ros::Time start, stop;
    start = ros::Time::now();

    // keep original image
    cv::Mat matrix;
    cv::Mat matrix_crop;  // high resolution
    // for painting
    cv::Mat matrix2;
    cv::Mat matrix2_crop;  // high resolution
    bool get_timestamp = false;
    ros::Time msgs_timestamp;
    bool has_crop_image = !crop_image_cache.empty();
    std::vector<msgs::PedObject> pedObjs;
    std::vector<msgs::DetectedObject> alertObjs;
    pedObjs.reserve(msg->objects.end() - msg->objects.begin());

    for (auto const& obj : msg->objects)
    {
      if (obj.classId != 1 || too_far(obj.bPoint))
      {  // 1 for people
        continue;
      }

      bool in_crop_range = false;
      if (obj.camInfo.v >= 692 && obj.camInfo.v + obj.camInfo.height < 1006 && has_crop_image && from_camera == 0)
      {
        in_crop_range = true;
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

      // Check object source is camera
      if (obj_pub.camInfo.width == 0 || obj_pub.camInfo.height == 0)
      {
        continue;
      }

      // Only first object need to check raw image
      if (!get_timestamp)
      {
        if (obj.header.stamp.toSec() > 1)
        {
          msgs_timestamp = obj.header.stamp;
        }
        else
        {
          msgs_timestamp = msg->header.stamp;
        }
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
        if (from_camera == 0)
        {
          for (int i = crop_image_cache.size() - 1; i >= 0; i--)
          {
            if (crop_image_cache[i].first <= msgs_timestamp || i == 0)
            {
#if PRINT_MESSAGE
              std::cout << "GOT CHA !!!!! time: " << crop_image_cache[i].first << " , " << msgs_timestamp << "crop"
                        << std::endl;
#endif

              matrix_crop = crop_image_cache[i].second;
              // for drawing bbox and keypoints
              matrix_crop.copyTo(matrix2_crop);
              get_timestamp = true;
              break;
            }
          }
        }
      }

      cv::Mat cropedImage;
      if (!in_crop_range)
      {
        // resize from 1920*1208 to 608*384
        obj_pub.camInfo.u *= scaling_ratio_width;
        obj_pub.camInfo.v *= scaling_ratio_height;
        obj_pub.camInfo.width *= scaling_ratio_width;
        obj_pub.camInfo.height *= scaling_ratio_height;
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

#if PRINT_MESSAGE
        std::cout << matrix.cols << " " << matrix.rows << " " << obj_pub.camInfo.u << " " << obj_pub.camInfo.v << " "
                  << obj_pub.camInfo.u + obj_pub.camInfo.width << " " << obj_pub.camInfo.v + obj_pub.camInfo.height
                  << std::endl;
#endif
        // crop image for openpose
        matrix.copyTo(cropedImage);
        cropedImage =
            cropedImage(cv::Rect(obj_pub.camInfo.u, obj_pub.camInfo.v, obj_pub.camInfo.width, obj_pub.camInfo.height));
      }
      else
      {
        obj_pub.camInfo.v -= 692;
#if PRINT_MESSAGE
        std::cout << matrix_crop.cols << " " << matrix_crop.rows << " " << obj_pub.camInfo.u << " " << obj_pub.camInfo.v
                  << " " << obj_pub.camInfo.u + obj_pub.camInfo.width << " "
                  << obj_pub.camInfo.v + obj_pub.camInfo.height << "crop" << std::endl;
#endif
        matrix_crop.copyTo(cropedImage);
        cropedImage =
            cropedImage(cv::Rect(obj_pub.camInfo.u, obj_pub.camInfo.v, obj_pub.camInfo.width, obj_pub.camInfo.height));

        obj_pub.camInfo.v += 692;
        // resize from 1920*1208 to 608*384
        obj_pub.camInfo.u *= scaling_ratio_width;
        obj_pub.camInfo.v *= scaling_ratio_height;
        obj_pub.camInfo.width *= scaling_ratio_width;
        obj_pub.camInfo.height *= scaling_ratio_height;

        // Avoid index out of bounds
        if (obj_pub.camInfo.u + obj_pub.camInfo.width > matrix.cols)
        {
          obj_pub.camInfo.width = matrix.cols - obj_pub.camInfo.u;
        }
        if (obj_pub.camInfo.v + obj_pub.camInfo.height > matrix.rows)
        {
          obj_pub.camInfo.height = matrix.rows - obj_pub.camInfo.v;
        }
      }

      // set size to resize cropped image for openpose
      // max pixel of width or height can only be 368
      int max_pixel = 368;
      float aspect_ratio = 0.0;
      int resize_height_to = 0;
      int resize_width_to = 0;
      if (cropedImage.cols >= cropedImage.rows)
      {  // width larger than height
        if (cropedImage.cols > max_pixel)
        {
          resize_width_to = max_pixel;
        }
        else
        {
          resize_width_to = cropedImage.cols;
        }
        resize_width_to = max_pixel;  // force to max pixel
        aspect_ratio = (float)cropedImage.rows / (float)cropedImage.cols;
        resize_height_to = int(aspect_ratio * resize_width_to);
      }
      else
      {  // height larger than width
        if (cropedImage.rows > max_pixel)
        {
          resize_height_to = max_pixel;
        }
        else
        {
          resize_height_to = cropedImage.rows;
        }
        resize_height_to = max_pixel;  // force to max pixel
        aspect_ratio = (float)cropedImage.cols / (float)cropedImage.rows;
        resize_width_to = int(aspect_ratio * resize_height_to);
      }
      cv::resize(cropedImage, cropedImage, cv::Size(resize_width_to, resize_height_to));

      std::vector<cv::Point2f> keypoints = get_openpose_keypoint(cropedImage);

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
        obj_pub.crossProbability =
            crossing_predict(obj.camInfo.u, obj.camInfo.v, obj.camInfo.u + obj.camInfo.width,
                             obj.camInfo.v + obj.camInfo.height, keypoints, obj.track.id, msg->header.stamp, buffer);
      }
      else
      {
        /*
                  std::vector<cv::Point2f> no_keypoint;
                  obj_pub.crossProbability =
                      crossing_predict(obj.camInfo.u, obj.camInfo.v, obj.camInfo.u + obj.camInfo.width,
                                        obj.camInfo.v + obj.camInfo.height, no_keypoint, obj.track.id,
            msg->header.stamp);
        */
      }
      obj_pub.facing_direction = get_facing_direction(keypoints);
      // obj_pub.body_direction = get_body_direction(keypoints);
      /*
            cv::Point p;
            p.x = 303;
            p.y = 275;
            cv::Point p2;
            p2.x = 303;
            p2.y = 383;
            cv::line(matrix2, p, p2, cv::Scalar(0, 0, 255), 1);
            cv::Point p3;
            p3.x = 403;
            p3.y = 383;
            cv::line(matrix2, p, p3, cv::Scalar(0, 0, 200), 1);
            cv::Point p4;
            p4.x = 503;
            p4.y = 383;
            cv::line(matrix2, p, p4, cv::Scalar(0, 0, 155), 1);
            cv::Point p5;
            p5.x = 603;
            p5.y = 383;
            cv::line(matrix2, p, p5, cv::Scalar(0, 0, 200), 1);
            cv::Point p6;
            p6.x = 203;
            p6.y = 383;
            cv::line(matrix2, p, p6, cv::Scalar(0, 0, 155), 1);
            cv::Point p7;
            p7.x = 53;
            p7.y = 383;
            cv::line(matrix2, p, p7, cv::Scalar(0, 0, 200), 1);
            cv::Point p8;
            p8.x = -247;
            p8.y = 383;
            cv::line(matrix2, p, p8, cv::Scalar(0, 0, 155), 1);
      */

      // only for front camera
      if (from_camera == 0)
      {
        obj_pub.crossProbability = adjust_probability(obj_pub);
      }

      if (use_2d_for_alarm)
      {
        // copy another nav_path to prevent vector changing while calculating
        std::vector<geometry_msgs::PoseStamped> nav_path_temp(nav_path);
        if (obj_pub.crossProbability * 100 >= cross_threshold)
        {
          if (obj_pub.bPoint.p0.x != 0 || obj_pub.bPoint.p0.y != 0)
          {
            msgs::PointXYZ camera_position = obj_pub.bPoint.p0;

            geometry_msgs::TransformStamped transform_stamped;
            try
            {
              transform_stamped = tfBuffer.lookupTransform("base_link", "map", ros::Time(0));
#if PRINT_MESSAGE
              std::cout << transform_stamped << std::endl;
#endif
            }
            catch (tf2::TransformException& ex)
            {
              ROS_WARN("%s", ex.what());
              ros::Duration(1.0).sleep();
              continue;
            }
            double yaw =
                get_tranform_yaw(transform_stamped.transform.rotation.x, transform_stamped.transform.rotation.y,
                                 transform_stamped.transform.rotation.z, transform_stamped.transform.rotation.w);

            // find the nearest nav_path point from pedestian's position
            geometry_msgs::PoseStamped nearest_point;
            double min_distance_from_path = 100000;
            for (const geometry_msgs::PoseStamped& path_point : nav_path_temp)
            {
              // coordinate transform for  nav_path (map) to camera (base_link)
              geometry_msgs::PoseStamped point_out;
              point_out.pose.position =
                  get_transform_coordinate(path_point.pose.position, yaw, transform_stamped.transform.translation);

              // calculate distance between pedestrian and each nav_path point
              double distance_diff = get_distance2(point_out.pose.position.x, point_out.pose.position.y,
                                                   camera_position.x, camera_position.y);
              if (min_distance_from_path > distance_diff)
              {
                min_distance_from_path = distance_diff;
                nearest_point = point_out;
              }
              nav_path_transformed.push_back(point_out);
            }
            double distance_from_car = 0;
            geometry_msgs::PoseStamped previous_path_point;
            bool passed_car_head = false;
            for (const geometry_msgs::PoseStamped& path_point : nav_path_transformed)
            {
              // check
              if (path_point.pose.position.x > 0)
              {
                passed_car_head = true;
              }
              // add distance between points
              if (passed_car_head)
              {
                distance_from_car +=
                    get_distance2(path_point.pose.position.x, path_point.pose.position.y,
                                  previous_path_point.pose.position.x, previous_path_point.pose.position.y);
              }
              if (path_point.pose.position.x == nearest_point.pose.position.x &&
                  path_point.pose.position.y == nearest_point.pose.position.y)
              {
#if DUMP_LOG
                // print distance
                file << ros::Time::now() << "," << obj_pub.track.id << "," << distance_from_car << ","
                     << veh_info.ego_speed << "\n";
#endif
#if PRINT_MESSAGE
                std::cout << "same, distance: " << distance_from_car << " id: " << obj_pub.track.id
                          << " time: " << ros::Time::now() << " speed: " << veh_info.ego_speed << std::endl;
#endif
                break;
              }
              previous_path_point = path_point;
            }
            // to free memory from vector
            nav_path_transformed.erase(nav_path_transformed.begin(), nav_path_transformed.end());

            double diff_x = (nearest_point.pose.position.x - camera_position.x) / 10;
            double diff_y = (nearest_point.pose.position.y - camera_position.y) / 10;
            alert_obj.track.forecasts.reserve(20);
            obj_pub.track.forecasts.reserve(20);
            alert_obj.track.is_ready_prediction = 1;
            obj_pub.track.is_ready_prediction = 1;
            for (int i = 0; i < 20; i++)
            {
              msgs::PathPrediction pp;
              pp.position.x = camera_position.x + diff_x * i;
              pp.position.y = camera_position.y + diff_y * i;
#if PRINT_MESSAGE
              std::cout << pp.position << std::endl;
#endif
              alert_obj.track.forecasts.push_back(pp);
              obj_pub.track.forecasts.push_back(pp);
            }
            alertObjs.push_back(alert_obj);
          }
        }
      }
      else
      {
        // copy another nav_path to prevent vector changing while calculating
        if (obj_pub.bPoint.p0.x != 0 || obj_pub.bPoint.p0.y != 0)
        {
          msgs::PointXYZ camera_position = obj_pub.bPoint.p0;
          std::vector<geometry_msgs::PoseStamped> nav_path_temp(nav_path);
          geometry_msgs::TransformStamped transform_stamped;
          try
          {
            transform_stamped = tfBuffer.lookupTransform("base_link", "map", ros::Time(0));
#if PRINT_MESSAGE
            std::cout << transform_stamped << std::endl;
#endif
          }
          catch (tf2::TransformException& ex)
          {
            ROS_WARN("%s", ex.what());
            ros::Duration(1.0).sleep();
            continue;
          }
          double yaw = get_tranform_yaw(transform_stamped.transform.rotation.x, transform_stamped.transform.rotation.y,
                                        transform_stamped.transform.rotation.z, transform_stamped.transform.rotation.w);

          // find the nearest nav_path point from pedestian's position
          geometry_msgs::PoseStamped nearest_point;
          double min_distance_from_path = 100000;
          for (const geometry_msgs::PoseStamped& path_point : nav_path_temp)
          {
            // coordinate transform for  nav_path (map) to camera (base_link)
            geometry_msgs::PoseStamped point_out;
            point_out.pose.position =
                get_transform_coordinate(path_point.pose.position, yaw, transform_stamped.transform.translation);

            // calculate distance between pedestrian and each nav_path point
            double distance_diff = get_distance2(point_out.pose.position.x, point_out.pose.position.y,
                                                 camera_position.x, camera_position.y);
            if (min_distance_from_path > distance_diff)
            {
              min_distance_from_path = distance_diff;
              nearest_point = point_out;
            }
            nav_path_transformed.push_back(point_out);
          }
          // too close to planned path
          // from center to left and right 2 meters
          if (min_distance_from_path < danger_zone_distance)
          {
            obj_pub.crossProbability = 1;
          }
          if (obj_pub.crossProbability * 100 >= cross_threshold)
          {
            double distance_from_car = 0;
            geometry_msgs::PoseStamped previous_path_point;
            bool passed_car_head = false;
            for (const geometry_msgs::PoseStamped& path_point : nav_path_transformed)
            {
              // check
              if (path_point.pose.position.x > 0)
              {
                passed_car_head = true;
              }
              // add distance between points
              if (passed_car_head)
              {
                distance_from_car +=
                    get_distance2(path_point.pose.position.x, path_point.pose.position.y,
                                  previous_path_point.pose.position.x, previous_path_point.pose.position.y);
              }
              if (path_point.pose.position.x == nearest_point.pose.position.x &&
                  path_point.pose.position.y == nearest_point.pose.position.y)
              {
#if DUMP_LOG
                // print distance
                file << ros::Time::now() << "," << obj_pub.track.id << "," << distance_from_car << ","
                     << veh_info.ego_speed << "\n";
#endif
#if PRINT_MESSAGE
                std::cout << "same, distance: " << distance_from_car << " id: " << obj_pub.track.id
                          << " time: " << ros::Time::now() << " speed: " << veh_info.ego_speed << std::endl;
#endif
                break;
              }
              previous_path_point = path_point;
            }
            // to free memory from vector
            nav_path_transformed.erase(nav_path_transformed.begin(), nav_path_transformed.end());

            double diff_x = (nearest_point.pose.position.x - camera_position.x) / 10;
            double diff_y = (nearest_point.pose.position.y - camera_position.y) / 10;
            alert_obj.track.forecasts.reserve(20);
            obj_pub.track.forecasts.reserve(20);
            alert_obj.track.is_ready_prediction = 1;
            obj_pub.track.is_ready_prediction = 1;
            for (int i = 0; i < 20; i++)
            {
              msgs::PathPrediction pp;
              pp.position.x = camera_position.x + diff_x * i;
              pp.position.y = camera_position.y + diff_y * i;
#if PRINT_MESSAGE
              std::cout << pp.position << std::endl;
#endif
              alert_obj.track.forecasts.push_back(pp);
              obj_pub.track.forecasts.push_back(pp);
            }
            alertObjs.push_back(alert_obj);
          }
        }
      }

      if (obj_pub.bPoint.p0.x != 0 || obj_pub.bPoint.p0.y != 0)
      {
        for (auto point : keypoints)
        {
          msgs::Keypoint kp;
          kp.x = point.x;
          kp.y = point.y;
          obj_pub.keypoints.push_back(kp);
        }
        pedObjs.push_back(obj_pub);
      }
    }

    msgs::DetectedObjectArray alert_objs;

    alert_objs.header = msg->header;
    alert_objs.header.frame_id = msg->header.frame_id;
    alert_objs.header.stamp = msg->header.stamp;
    alert_objs.objects.assign(alertObjs.begin(), alertObjs.end());
    alert_pub.publish(alert_objs);

    msgs::PedObjectArray msg_pub;
    // msg_pub.raw_image = img_msg;
    msg_pub.header = msg->header;
    msg_pub.header.frame_id = msg->header.frame_id;
    msg_pub.header.stamp = msg->header.stamp;
    msg_pub.objects.assign(pedObjs.begin(), pedObjs.end());
    if (from_camera == 0) // front
    {
      chatter_pub_front.publish(msg_pub);
    }
    else if (from_camera == 1) // left
    {
      chatter_pub_left.publish(msg_pub);
    }
    else if (from_camera == 2) // right
    {
      chatter_pub_right.publish(msg_pub);
    }
    delay_from_camera = std::to_string((ros::Time::now() - msgs_timestamp).toSec());

    matrix2 = 0;

    stop = ros::Time::now();
    total_time += stop - start;
    chatter_callback_info = "Cost time: " + std::to_string((stop - start).toSec()) +
                            "(sec) Total cost time: " + std::to_string((total_time).toSec()) +
                            "(sec) Loop: " + std::to_string(count);
#if PRINT_MESSAGE
    std::cout << "Camera source: " << from_camera << std::endl;
    std::cout << "Cost time: " << stop - start << " sec" << std::endl;
    std::cout << "Total time: " << total_time << " sec / loop: " << count << std::endl;
#endif
  }
}

geometry_msgs::Point PedestrianEvent::get_transform_coordinate(geometry_msgs::Point origin_point, double yaw,
                                                               geometry_msgs::Vector3 translation)
{
  geometry_msgs::Point new_point;
  new_point.x = translation.x + std::cos(yaw) * origin_point.x - std::sin(yaw) * origin_point.y;
  new_point.y = translation.y + std::sin(yaw) * origin_point.x + std::cos(yaw) * origin_point.y;
  return new_point;
}

double PedestrianEvent::get_tranform_yaw(double x, double y, double z, double w)
{
  double roll, pitch, yaw;

  tf::Quaternion q(x, y, z, w);
  tf::Matrix3x3 m(q);
  m.getRPY(roll, pitch, yaw);

  return yaw;
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
  // pedestrian in danger zone, force determine as Cross
  if (std::fabs(x - 303) < 100 * (y - 275) / 108)
  {
    if (use_2d_for_alarm)
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
  // if (std::fabs(x-303)<300*(y-275)/108)
  // {
  // return obj.crossProbability*0.7;
  // }
  // std::fabs(x-303)>=300*(y-275)/108
  return obj.crossProbability * 0.7;
}

void PedestrianEvent::draw_ped_front_callback(const msgs::PedObjectArray::ConstPtr& msg)
{
  draw_pedestrians_callback(msg, front_image_cache, 0);
}

void PedestrianEvent::draw_ped_left_callback(const msgs::PedObjectArray::ConstPtr& msg)
{
  draw_pedestrians_callback(msg, left_image_cache, 1);
}

void PedestrianEvent::draw_ped_right_callback(const msgs::PedObjectArray::ConstPtr& msg)
{
  draw_pedestrians_callback(msg, right_image_cache, 2);
}

void PedestrianEvent::draw_pedestrians_callback(const msgs::PedObjectArray::ConstPtr& msg, boost::circular_buffer<std::pair<ros::Time, cv::Mat>> &image_cache, int from_camera)
{
  ped_info.clear();
  if (image_cache.empty())  // do if there is image in buffer
  {
    return;
  }

  cv::Mat matrix;
  cv::Mat matrix2;
  ros::Time msgs_timestamp = ros::Time(0);
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
  else
  {
    msgs_timestamp = msg->header.stamp;
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

  for (unsigned int i = 0; i < msg->objects.size(); i++)
  {
    auto const& obj = msg->objects[i];
    std::vector<cv::Point2f> keypoints;
    int keypoint_number = 0;
    for (auto const& point : msg->objects[i].keypoints)
    {
      cv::Point2f kp;
      kp.x = point.x;
      kp.y = point.y;
      keypoints.push_back(kp);
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

    cv::Rect box;
    box.x = obj.camInfo.u;
    box.y = obj.camInfo.v;
    box.width = obj.camInfo.width;
    box.height = obj.camInfo.height;
    cv::rectangle(matrix, box.tl(), box.br(), CV_RGB(0, 255, 0), 1);
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
    if (p >= cross_threshold)
    {
      if (show_probability)
      {
        probability = "C(" + std::to_string(p / 100) + "." + std::to_string(p / 10 % 10) + std::to_string(p % 10) + ")";
      }
      else
      {
        probability = "C";
      }

      cv::putText(matrix, probability, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1 /*font size*/, cv::Scalar(0, 50, 255), 2,
                  4, 0);
    }
    else
    {
      if (show_probability)
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
                  4, 0);
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
    // cv::putText(matrix, id_print, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5 /*font size*/, cv::Scalar(100, 220, 0),
    // 2,
    // 4,0);

    // box.x -= 0;
    // draw face direction
    if (obj.facing_direction == 0)
    {
      id_print += "left ";
      // facing left hand side
      // cv::putText(matrix, "<-", box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1 /*font size*/, cv::Scalar(100, 220, 0), 2,
      // 4,
      // 0);
    }
    else if (obj.facing_direction == 1)
    {
      id_print += "right";
      // facing right hand side
      // cv::putText(matrix, "->", box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1 /*font size*/, cv::Scalar(100, 220, 0), 2,
      // 4,
      // 0);
    }
    else if (obj.facing_direction == 2)
    {
      id_print += "back ";
      // facing car side
      // cv::putText(matrix, "O", box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1 /*font size*/, cv::Scalar(100, 220, 0), 2, 4,
      // 0);
    }
    else
    {
      id_print += "front";
      // facing car opposite side
      // cv::putText(matrix, "X", box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1 /*font size*/, cv::Scalar(100, 220, 0), 2, 4,
      // 0);
    }
    cv::putText(matrix, id_print, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5 /*font size*/, cv::Scalar(100, 220, 0), 1, 2,
                0);

    //   cv::Rect box2 = box;
    //   box.y += 30;
    //   box2.y += 30;
    //   box2.width = 0;
    //   // draw left leg direction
    //   if (obj.body_direction / 10 == 0)
    //   {
    //     // facing left hand side
    //     cv::putText(matrix, "<-", box2.br(), cv::FONT_HERSHEY_SIMPLEX, 0.5 /*font size*/, cv::Scalar(100, 220, 0), 2,
    //     4,
    //                 0);
    //   }
    //   else if (obj.body_direction / 10 == 1)
    //   {
    //     // facing right hand side
    //     cv::putText(matrix, "->", box2.br(), cv::FONT_HERSHEY_SIMPLEX, 0.5 /*font size*/, cv::Scalar(100, 220, 0), 2,
    //     4,
    //                 0);
    //   }
    //   else if (obj.body_direction / 10 == 2)
    //   {
    //     // facing car side
    //     cv::putText(matrix, "O", box2.br(), cv::FONT_HERSHEY_SIMPLEX, 0.5 /*font size*/, cv::Scalar(100, 220, 0), 2,
    //     4,
    //                 0);
    //   }
    //   else
    //   {
    //     // facing car opposite side
    //     cv::putText(matrix, "X", box2.br(), cv::FONT_HERSHEY_SIMPLEX, 0.5 /*font size*/, cv::Scalar(100, 220, 0), 2,
    //     4,
    //                 0);
    //   }

    //   // draw right leg direction
    //   if (obj.body_direction % 10 == 0)
    //   {
    //     // facing left hand side
    //     cv::putText(matrix, "<-", box.br(), cv::FONT_HERSHEY_SIMPLEX, 0.5 /*font size*/, cv::Scalar(100, 220, 0), 2,
    //     4,
    //                 0);
    //   }
    //   else if (obj.body_direction % 10 == 1)
    //   {
    //     // facing right hand side
    //     cv::putText(matrix, "->", box.br(), cv::FONT_HERSHEY_SIMPLEX, 0.5 /*font size*/, cv::Scalar(100, 220, 0), 2,
    //     4,
    //                 0);
    //   }
    //   else if (obj.body_direction % 10 == 2)
    //   {
    //     // facing car side
    //     cv::putText(matrix, "O", box.br(), cv::FONT_HERSHEY_SIMPLEX, 0.5 /*font size*/, cv::Scalar(100, 220, 0), 2,
    //     4, 0);
    //   }
    //   else
    //   {
    //     // facing car opposite side
    //     cv::putText(matrix, "X", box.br(), cv::FONT_HERSHEY_SIMPLEX, 0.5 /*font size*/, cv::Scalar(100, 220, 0), 2,
    //     4, 0);
    //   }
    ped_info.push_back(id_print + " " + probability + " x: " + std::to_string((int)obj.bPoint.p0.x) + " y: " +
                       std::to_string((int)obj.bPoint.p0.y) + " keypoints number: " + std::to_string(keypoint_number));
  }
  // do resize only when computer cannot support
  // cv::resize(matrix, matrix, cv::Size(matrix.cols / 1, matrix.rows / 1));

  // make cv::Mat to sensor_msgs::Image
  sensor_msgs::ImageConstPtr msg_pub2 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", matrix).toImageMsg();

  if (from_camera == 0) // front
  {
    box_pub_front.publish(msg_pub2);
  }
  else if (from_camera == 1) // left
  {
    box_pub_left.publish(msg_pub2);
  }
  else if (from_camera == 2) // right
  {
    box_pub_right.publish(msg_pub2);
  }

  matrix = 0;
}

/**
 * return
 * 0 for facing left
 * 1 for facing right
 * 2 for facing car side
 * 3 for facing car opposite side
 */
int PedestrianEvent::get_facing_direction(const std::vector<cv::Point2f>& keypoints)
{
  bool look_at_left = false;
  bool look_at_right = false;
  bool only_left_ear = false;
  bool only_right_ear = false;
  // if no left eye but left ear is detected
  if (!keypoint_is_detected(keypoints.at(16)) && keypoint_is_detected(keypoints.at(18)))
  {
    only_left_ear = true;
  }
  // if no right eye but right ear is detected
  if (!keypoint_is_detected(keypoints.at(15)) && keypoint_is_detected(keypoints.at(17)))
  {
    only_right_ear = true;
  }
  // if no eye detected
  if (only_left_ear || only_right_ear)
  {
    if (only_left_ear && !only_right_ear)
    {
      // if only left ear
      return 0;
    }
    if (!only_left_ear && only_right_ear)
    {
      // if only right ear
      return 1;
    }
    if (only_left_ear && only_right_ear)
    {
      // if both ears detected
      if (keypoints.at(17).x > keypoints.at(18).x)
      {
        // if right ear is on the right side of left ear
        // that is facing car opposite side
        return 3;
      }
      else
      {
        // if right ear is on the left side of left ear
        // that is facing car side
        return 2;
      }
    }
  }
  else
  {
    // if left ear and left eye are detected and left eye is on the left side of left ear
    if (keypoint_is_detected(keypoints.at(16)) && keypoint_is_detected(keypoints.at(18)) &&
        keypoints.at(16).x < keypoints.at(18).x)
    {
      look_at_left = true;
    }
    // if right ear and right eye are detected and right eye is on the right side of right ear
    if (keypoint_is_detected(keypoints.at(15)) && keypoint_is_detected(keypoints.at(17)) &&
        keypoints.at(15).x > keypoints.at(17).x)
    {
      look_at_right = true;
    }
  }

  if (look_at_left && !look_at_right)
  {
    // facing left hand side
    return 0;
  }
  else if (!look_at_left && look_at_right)
  {
    // facing right hand side
    return 1;
  }
  else if (look_at_left && look_at_right)
  {
    // facing car side
    return 2;
  }
  else
  {
    // facing car opposite side
    return 3;
  }
  // defalt: facing car opposite side
  return 3;
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
float PedestrianEvent::crossing_predict(float bb_x1, float bb_y1, float bb_x2, float bb_y2,
                                        std::vector<cv::Point2f> keypoint, int id, ros::Time time, Buffer &buffer)
{
  try
  {
    // initialize feature
    std::vector<float> feature;

    // Add bbox to feature vector
    float bbox[] = { bb_x1, bb_y1, bb_x2, bb_y2 };
    feature.insert(feature.end(), bbox, bbox + sizeof(bbox) / sizeof(bbox[0]));

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
    }
    else  // if keypoint is empty
    {
      float* zero_arr;
      // The first four feature are bb_x1, bb_y1, bb_x2, bb_y2
      int other_feature = feature_num - 4;
      zero_arr = new float[other_feature]();
      feature.insert(feature.begin(), zero_arr, zero_arr + sizeof(zero_arr) / sizeof(zero_arr[0]));
      delete[] zero_arr;
    }

    //  Buffer first frame
    if (buffer.timestamp == ros::Time(0))
    {
      buffer.timestamp = time;
    }
    //  new frame
    else if (buffer.timestamp != time)
    {
      buffer.timestamp = time;
      buffer.check_life();
    }
    feature = buffer.add(id, feature);

#if PRINT_MESSAGE
    buffer.display();
#endif
    // Convert vector to array
    int total_feature_size = feature_num * frame_num;
    float feature_arr[total_feature_size];
    std::copy(feature.begin(), feature.end(), feature_arr);
    // Convert array to Mat
    cv::Mat feature_mat = cv::Mat(1, total_feature_size, CV_32F, feature_arr);
    // Predict
    float predict_result = predict_rf_pose(feature_mat);

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
  return M_PI / 2 - atan2(std::fabs(y1 - y2), std::fabs(x1 - x2));
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
    angle[0] = acos((a * a + c * c - b * b) / (2 * a * c));
    angle[1] = acos((a * a + b * b - c * c) / (2 * a * b));
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
float PedestrianEvent::predict_rf_pose(cv::Mat input_data)
{
  cv::Mat votes;
  rf_pose->getVotes(input_data, votes, 0);
  float positive = votes.at<int>(1, 1);
  float negative = votes.at<int>(1, 0);
  float p = positive / (negative + positive);

#if PRINT_MESSAGE
  std::cout << "prediction: " << p << votes.size() << std::endl;
  std::cout << votes.at<int>(0, 0) << " " << votes.at<int>(0, 1) << std::endl;
  std::cout << votes.at<int>(1, 0) << " " << votes.at<int>(1, 1) << std::endl;
#endif

  return p;
}

bool PedestrianEvent::too_far(const msgs::BoxPoint box_point)
{
  if ((box_point.p0.x + box_point.p6.x) / 2 > max_distance)
  {
    return true;
  }
  else
  {
    return false;
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
  if (input_source == 0)
  {
    sub_1 = nh_sub_1.subscribe("/cam_obj/front_bottom_60", 1, &PedestrianEvent::front_callback,
                               this);  // /CamObjFrontCenter is sub topic
    sub_2 = nh_sub_2.subscribe("/cam/front_bottom_60", 1, &PedestrianEvent::cache_front_image_callback,
                               this);  // /cam/F_center is sub topic
    sub_3 = nh_sub_3.subscribe("/nav_path_astar_final", 1, &PedestrianEvent::nav_path_callback,
                               this);  // /cam/F_center is sub topic
    sub_4 = nh_sub_4.subscribe("/veh_info", 1, &PedestrianEvent::veh_info_callback,
                               this);  // /cam/F_center is sub topic
    sub_5 = nh_sub_5.subscribe("/cam/front_bottom_60_crop", 1, &PedestrianEvent::cache_crop_image_callback,
                               this);  // /cam/F_center is sub topic
    sub_6 = nh_sub_6.subscribe("/PedCross/Pedestrians", 1, &PedestrianEvent::draw_ped_front_callback,
                               this);  // /cam/F_center is sub topic
  }
  else if (input_source == 1)
  {
    sub_1 = nh_sub_1.subscribe("/cam_obj/left_back_60", 1, &PedestrianEvent::front_callback,
                               this);  // /CamObjFrontLeft is sub topic
    sub_2 = nh_sub_2.subscribe("/cam/left_back_60", 1, &PedestrianEvent::cache_front_image_callback,
                               this);  // /cam/F_left is sub topic
    sub_3 = nh_sub_3.subscribe("/nav_path_astar_final", 1, &PedestrianEvent::nav_path_callback,
                               this);  // /cam/F_center is sub topic
    sub_4 = nh_sub_4.subscribe("/veh_info", 1, &PedestrianEvent::veh_info_callback,
                               this);  // /cam/F_center is sub topic
    sub_5 = nh_sub_5.subscribe("/cam/front_bottom_60_crop", 1, &PedestrianEvent::cache_crop_image_callback,
                               this);  // /cam/F_center is sub topic
    sub_6 = nh_sub_6.subscribe("/PedCross/Pedestrians", 1, &PedestrianEvent::draw_ped_front_callback,
                               this);  // /cam/F_center is sub topic
  }
  else if (input_source == 2)
  {
    sub_1 = nh_sub_1.subscribe("/cam_obj/right_back_60", 1, &PedestrianEvent::front_callback,
                               this);  // /CamObjFrontRight is sub topic
    sub_2 = nh_sub_2.subscribe("/cam/right_back_60", 1, &PedestrianEvent::cache_front_image_callback,
                               this);  // /cam/F_right is sub topic
    sub_3 = nh_sub_3.subscribe("/nav_path_astar_final", 1, &PedestrianEvent::nav_path_callback,
                               this);  // /cam/F_center is sub topic
    sub_4 = nh_sub_4.subscribe("/veh_info", 1, &PedestrianEvent::veh_info_callback,
                               this);  // /cam/F_center is sub topic
    sub_5 = nh_sub_5.subscribe("/cam/front_bottom_60_crop", 1, &PedestrianEvent::cache_crop_image_callback,
                               this);  // /cam/F_center is sub topic
    sub_6 = nh_sub_6.subscribe("/PedCross/Pedestrians", 1, &PedestrianEvent::draw_ped_front_callback,
                               this);  // /cam/F_center is sub topic
  }
  else  if (input_source == 3)
  {
    sub_1 = nh_sub_1.subscribe("/Tracking2D", 1, &PedestrianEvent::front_callback,
                               this);  // /PathPredictionOutput is sub topic
    sub_2 = nh_sub_2.subscribe("/cam/front_bottom_60", 1, &PedestrianEvent::cache_front_image_callback,
                               this);  // /cam/F_right is sub topic
    sub_3 = nh_sub_3.subscribe("/nav_path_astar_final", 1, &PedestrianEvent::nav_path_callback,
                               this);  // /cam/F_center is sub topic
    sub_4 = nh_sub_4.subscribe("/veh_info", 1, &PedestrianEvent::veh_info_callback,
                               this);  // /cam/F_center is sub topic
    sub_5 = nh_sub_5.subscribe("/cam/front_bottom_60_crop", 1, &PedestrianEvent::cache_crop_image_callback,
                               this);  // /cam/F_center is sub topic
    sub_6 = nh_sub_6.subscribe("/PedCross/Pedestrians", 1, &PedestrianEvent::draw_ped_front_callback,
                               this);  // /cam/F_center is sub topic
  }
  else  // if (input_source == 4)
  {
    sub_1 = nh_sub_1.subscribe("/Tracking2D/front_bottom_60", 1, &PedestrianEvent::front_callback,
                               this);  // /Tracking2D/front_bottom_60 is subscirbe topic
    sub_2 = nh_sub_2.subscribe("/Tracking2D/left_back_60", 1, &PedestrianEvent::left_callback,
                               this);  // /Tracking2D/left_back_60 is subscirbe topic
    sub_3 = nh_sub_3.subscribe("/Tracking2D/right_back_60", 1, &PedestrianEvent::right_callback,
                               this);  // /Tracking2D/right_back_60 is subscirbe topic
    sub_4 = nh_sub_4.subscribe("/cam/front_bottom_60", 1, &PedestrianEvent::cache_front_image_callback,
                               this);  // /cam/F_right is subscirbe topic
    sub_5 = nh_sub_5.subscribe("/cam/front_bottom_60_crop", 1, &PedestrianEvent::cache_crop_image_callback,
                               this);  // /cam/F_center is subscirbe topic
    sub_6 = nh_sub_6.subscribe("/cam/left_back_60", 1, &PedestrianEvent::cache_left_image_callback,
                               this);  // /cam/F_center is subscirbe topic
    sub_7 = nh_sub_7.subscribe("/cam/right_back_60", 1, &PedestrianEvent::cache_right_image_callback,
                               this);  // /cam/F_center is subscirbe topic
    sub_8 = nh_sub_8.subscribe("/nav_path_astar_final", 1, &PedestrianEvent::nav_path_callback,
                               this);  // /cam/F_center is subscirbe topic
    sub_9 = nh_sub_9.subscribe("/veh_info", 1, &PedestrianEvent::veh_info_callback,
                               this);  // /cam/F_center is subscirbe topic
    sub_10 = nh_sub_10.subscribe("/PedCross/Pedestrians/front_bottom_60", 1, &PedestrianEvent::draw_ped_front_callback,
                               this);  // /cam/F_center is subscirbe topic
    sub_11 = nh_sub_11.subscribe("/PedCross/Pedestrians/left_back_60", 1, &PedestrianEvent::draw_ped_front_callback,
                               this);  // /cam/F_center is subscirbe topic
    sub_12 = nh_sub_12.subscribe("/PedCross/Pedestrians/right_back_60", 1, &PedestrianEvent::draw_ped_front_callback,
                               this);  // /cam/F_center is subscirbe topic
  }

  // Create AsyncSpinner, run it on all available cores and make it process custom callback queue
  g_spinner_1.reset(new ros::AsyncSpinner(0, &queue_1));
  g_spinner_2.reset(new ros::AsyncSpinner(0, &queue_2));
  g_spinner_3.reset(new ros::AsyncSpinner(0, &queue_3));
  g_spinner_4.reset(new ros::AsyncSpinner(0, &queue_4));
  g_spinner_5.reset(new ros::AsyncSpinner(0, &queue_5));
  g_spinner_6.reset(new ros::AsyncSpinner(0, &queue_6));
  g_spinner_7.reset(new ros::AsyncSpinner(0, &queue_7));
  g_spinner_8.reset(new ros::AsyncSpinner(0, &queue_8));
  g_spinner_9.reset(new ros::AsyncSpinner(0, &queue_9));
  g_spinner_10.reset(new ros::AsyncSpinner(0, &queue_10));
  g_spinner_11.reset(new ros::AsyncSpinner(0, &queue_11));

  g_enable = true;
  g_trigger = true;

  // Loop with 100 Hz rate
  ros::Rate loop_rate(20);
  while (ros::ok())
  {
    // Enable state changed
    if (g_trigger)
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
      // Start the spinner
      g_spinner_1->start();
      g_spinner_2->start();
      g_spinner_3->start();
      g_spinner_4->start();
      g_spinner_5->start();
      g_spinner_6->start();
      g_spinner_7->start();
      g_spinner_8->start();
      g_spinner_9->start();
      g_spinner_10->start();
      g_spinner_11->start();
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
  g_spinner_4.reset();
  g_spinner_5.reset();
  g_spinner_6.reset();
  g_spinner_7.reset();
  g_spinner_8.reset();
  g_spinner_9.reset();
  g_spinner_10.reset();
  g_spinner_11.reset();
  // Wait for ROS threads to terminate
  ros::waitForShutdown();
}

// return 25 keypoints detected by openpose
std::vector<cv::Point2f> PedestrianEvent::get_openpose_keypoint(cv::Mat input_image)
{
#if PRINT_MESSAGE
  ros::Time timer = ros::Time::now();
#endif

  std::vector<cv::Point2f> points;
  points.reserve(number_keypoints * 2);

  float height = input_image.rows;

  std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datumToProcess = createDatum(input_image);
  bool successfullyEmplaced = openPose.waitAndEmplace(datumToProcess);
  std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datumProcessed;
  if (successfullyEmplaced && openPose.waitAndPop(datumProcessed))
  {
    // display(datumProcessed);
    if (datumProcessed != nullptr && !datumProcessed->empty())
    {
#if PRINT_MESSAGE
      op::opLog("\nKeypoints:");
#endif
      // Accesing each element of the keypoints
      const auto& poseKeypoints = datumProcessed->at(0)->poseKeypoints;
#if PRINT_MESSAGE
      op::opLog("Person pose keypoints:");
#endif
      for (auto person = 0; person < poseKeypoints.getSize(0); person++)
      {
#if PRINT_MESSAGE
        op::opLog("Person " + std::to_string(person) + " (x, y, score):");
#endif
        for (auto bodyPart = 0; bodyPart < poseKeypoints.getSize(1); bodyPart++)
        {
          float x = poseKeypoints[{ person, bodyPart, 0 }] / height;
          float y = poseKeypoints[{ person, bodyPart, 1 }] / height;
          points.push_back(cv::Point2f(x, y));

          std::string valueToPrint;
          for (auto xyscore = 0; xyscore < poseKeypoints.getSize(2); xyscore++)
          {
            valueToPrint += std::to_string(poseKeypoints[{ person, bodyPart, xyscore }]) + " ";
          }
#if PRINT_MESSAGE
          op::opLog(valueToPrint);
#endif
        }
      }
    }
  }

#if PRINT_MESSAGE
  std::cout << "Openpose time cost: " << ros::Time::now() - timer << std::endl;
#endif

  for (int i = 0; i < 25; i++)
  {
    points.push_back(cv::Point2f(0.0, 0.0));
  }

  return points;
}

bool PedestrianEvent::display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
  // User's displaying/saving/other processing here
  // datum.cvOutputData: rendered frame with pose or heatmaps
  // datum.poseKeypoints: Array<float> with the estimated pose
  char key = ' ';
  if (datumsPtr != nullptr && !datumsPtr->empty())
  {
    cv::imshow("User worker GUI", OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData));
    // Display image and sleeps at least 1 ms (it usually sleeps ~5-10 msec to display the image)
    key = (char)cv::waitKey(1);
  }
  else
  {
    op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
  }
  return (key == 27);
}

std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> PedestrianEvent::createDatum(cv::Mat mat)
{
  // Create new datum
  auto datumsPtr = std::make_shared<std::vector<std::shared_ptr<op::Datum>>>();
  datumsPtr->emplace_back();
  auto& datumPtr = datumsPtr->at(0);
  datumPtr = std::make_shared<op::Datum>();

  // Fill datum
  datumPtr->cvInputData = OP_CV2OPCONSTMAT(mat);

  return datumsPtr;
}

int PedestrianEvent::openPoseROS()
{
  // logging_level
  op::checkBool(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__,
                __FUNCTION__, __FILE__);
  op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
  op::Profiler::setDefaultX(FLAGS_profile_speed);

  op::opLog("Starting pose estimation thread(s)", op::Priority::High);
  openPose.start();

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
  tf2_ros::TransformListener tfListener(pe.tfBuffer);
std::cout<< PED_MODEL_DIR + std::string("/rf_10frames_normalization_15peek.yml") << std::endl;
  pe.rf_pose = cv::ml::StatModel::load<cv::ml::RTrees>(PED_MODEL_DIR + std::string("/rf_10frames_normalization_15peek.yml"));
  
  ros::NodeHandle nh1;
  pe.chatter_pub_front = nh1.advertise<msgs::PedObjectArray>("/PedCross/Pedestrians/front_bottom_60", 1);  // /PedCross/Pedestrians is pub topic
  ros::NodeHandle nh2;
  pe.box_pub_front = nh2.advertise<sensor_msgs::Image&>("/PedCross/DrawBBox/front_bottom_60", 1);  // /PedCross/DrawBBox is pub topic
  ros::NodeHandle nh3;
  pe.chatter_pub_left = nh3.advertise<msgs::PedObjectArray>("/PedCross/Pedestrians/left_back_60", 1);  // /PedCross/Pedestrians is pub topic
  ros::NodeHandle nh4;
  pe.box_pub_left = nh4.advertise<sensor_msgs::Image&>("/PedCross/DrawBBox/left_back_60", 1);  // /PedCross/DrawBBox is pub topic
  ros::NodeHandle nh5;
  pe.chatter_pub_right = nh5.advertise<msgs::PedObjectArray>("/PedCross/Pedestrians/right_back_60", 1);  // /PedCross/Pedestrians is pub topic
  ros::NodeHandle nh6;
  pe.box_pub_right = nh6.advertise<sensor_msgs::Image&>("/PedCross/DrawBBox/right_back_60", 1);  // /PedCross/DrawBBox is pub topic
  ros::NodeHandle nh7;
  pe.alert_pub = nh7.advertise<msgs::DetectedObjectArray>("/PedCross/Alert", 1);  // /PedCross/DrawBBox is pub topic
  // Get parameters from ROS
  ros::NodeHandle nh;
  nh.param<bool>("/pedestrian_event/show_probability", pe.show_probability, true);
  nh.param<int>("/pedestrian_event/input_source", pe.input_source, 4);
  nh.param<double>("/pedestrian_event/max_distance", pe.max_distance, 50);
  nh.param<double>("/pedestrian_event/danger_zone_distance", pe.danger_zone_distance, 2);
  nh.param<bool>("/pedestrian_event/use_2d_for_alarm", pe.use_2d_for_alarm, false);

  pe.front_image_cache = boost::circular_buffer<std::pair<ros::Time, cv::Mat>>(pe.buffer_size);
  pe.left_image_cache = boost::circular_buffer<std::pair<ros::Time, cv::Mat>>(pe.buffer_size);
  pe.right_image_cache = boost::circular_buffer<std::pair<ros::Time, cv::Mat>>(pe.buffer_size);
  pe.crop_image_cache = boost::circular_buffer<std::pair<ros::Time, cv::Mat>>(pe.buffer_size);
  pe.buffer_front.initial();
  pe.buffer_left.initial();
  pe.buffer_right.initial();

  pe.openPoseROS();

  stop = ros::Time::now();
  std::cout << "PedCross started. Init time: " << stop - start << " sec" << std::endl;
  pe.count = 0;

  pe.nav_path_transformed.reserve(200);
  pe.nav_path.reserve(200);

  std::stringstream ss;
  ss << "../../../ped_output.csv";
  std::string fname = ss.str();
  pe.file.open(fname, std::ios_base::app);
  pe.run();
  pe.file.close();

  return 0;
}
