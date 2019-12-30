#include "pedestrian_event.h"
#if USE_GLOG
#include <glog/logging.h>
#endif

namespace ped
{
void PedestrianEvent::run()
{
  pedestrian_event();
}

void PedestrianEvent::cache_image_callback(const sensor_msgs::Image::ConstPtr& msg)
{
#if USE_GLOG
  ros::Time start;
  start = ros::Time::now();
#endif

  // buffer raw image in cv::Mat with timestamp
  cv_bridge::CvImageConstPtr cv_ptr_image;
  cv_ptr_image = cv_bridge::toCvShare(msg, "bgr8");
  cv::Mat msg_decode;
  cv_ptr_image->image.copyTo(msg_decode);

  // buffer raw image in msg
  imageCache.push_back({ msg->header.stamp, msg_decode });

#if USE_GLOG
  std::cout << "Image buffer time cost: " << ros::Time::now() - start << std::endl;
#endif
}

void PedestrianEvent::chatter_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  if (!imageCache.empty())  // do if there is image in buffer
  {
    count++;
#if USE_GLOG
    ros::Time start, stop;
    start = ros::Time::now();
// std::cout << "time stamp: " << msg->header.stamp << " buffer size: " << imageCache.size() << std::endl;
#endif

    // compare and get the raw image when object detected
    cv::Mat matrix;
    for (int i = imageCache.size() - 1; i >= 0; i--)
    {
      if (imageCache[i].first <= msg->header.stamp || i == 0)
      {
#if USE_GLOG
        std::cout << "GOT CHA !!!!! time: " << imageCache[i].first << " , " << msg->header.stamp << std::endl;
#endif

        matrix = imageCache[i].second;
        break;
      }
    }

    // for drawing bbox and keypoints
    cv::Mat matrix2;
    matrix.copyTo(matrix2);

    std::vector<msgs::PedObject> pedObjs;
    pedObjs.reserve(msg->objects.end() - msg->objects.begin());
    for (auto const& obj : msg->objects)
    {
      if (obj.classId == 1)  // 1 for people
      {
        if (too_far(obj.bPoint))
          continue;

        // set msg infomation
        msgs::PedObject obj_pub;
        obj_pub.header = obj.header;
        obj_pub.header.frame_id = obj.header.frame_id;
        obj_pub.header.stamp = obj.header.stamp;
        obj_pub.classId = obj.classId;
        obj_pub.camInfo = obj.camInfo;
        obj_pub.bPoint = obj.bPoint;
        obj_pub.track.id = obj.track.id;
#if USE_GLOG
        std::cout << "Track ID: " << obj.track.id << std::endl;
#endif
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

#if USE_GLOG
        std::cout << matrix.cols << " " << matrix.rows << " " << obj_pub.camInfo.u << " " << obj_pub.camInfo.v << " "
                  << obj_pub.camInfo.u + obj_pub.camInfo.width << " " << obj_pub.camInfo.v + obj_pub.camInfo.height
                  << std::endl;
#endif

        // crop image for openpose
        cv::Mat cropedImage =
            matrix(cv::Rect(obj_pub.camInfo.u, obj_pub.camInfo.v, obj_pub.camInfo.width, obj_pub.camInfo.height));
        // cv::imwrite( "/home/itri457854/frame2.png", cropedImage );

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

        sensor_msgs::ImageConstPtr msg_pub3 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cropedImage).toImageMsg();
        pose_pub.publish(msg_pub3);
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
              has_keypoint = true;
          }
        }
        if (has_keypoint)
          obj_pub.crossProbability =
              crossing_predict(obj.camInfo.u, obj.camInfo.v, obj.camInfo.u + obj.camInfo.width,
                               obj.camInfo.v + obj.camInfo.height, keypoints, obj.track.id, msg->header.stamp);
        else
        {
          std::vector<cv::Point2f> no_keypoint;
          obj_pub.crossProbability =
              crossing_predict(obj.camInfo.u, obj.camInfo.v, obj.camInfo.u + obj.camInfo.width,
                               obj.camInfo.v + obj.camInfo.height, no_keypoint, obj.track.id, msg->header.stamp);
        }
        pedObjs.push_back(obj_pub);

        // draw keypoints on cropped and whole image
        for (unsigned int i = 0; i < body_part_size; i++)
        {
          keypoints.at(body_part[i]).x = keypoints.at(body_part[i]).x * obj_pub.camInfo.height;
          keypoints.at(body_part[i]).y = keypoints.at(body_part[i]).y * obj_pub.camInfo.height;
          if (keypoints.at(body_part[i]).x != 0 || keypoints.at(body_part[i]).y != 0)
          {
            cv::circle(cropedImage, keypoints.at(body_part[i]), 2, cv::Scalar(0, 255, 0));
            cv::Point p = keypoints.at(body_part[i]);
            p.x = obj_pub.camInfo.u + p.x;
            p.y = obj_pub.camInfo.v + p.y;
            cv::circle(matrix2, p, 2, cv::Scalar(0, 255, 0), -1);
          }
        }
        // draw hands
        int body_part1[7] = { 4, 3, 2, 1, 5, 6, 7 };
        unsigned int body_part1_size = sizeof(body_part1) / sizeof(*body_part1);
        for (unsigned int i = 0; i < body_part1_size - 1; i++)
        {
          if ((keypoints.at(body_part1[i]).x != 0 || keypoints.at(body_part1[i]).y != 0) &&
              (keypoints.at(body_part1[i + 1]).x != 0 || keypoints.at(body_part1[i + 1]).y != 0))
          {
            cv::Point p = keypoints.at(body_part1[i]);
            p.x = obj_pub.camInfo.u + p.x;
            p.y = obj_pub.camInfo.v + p.y;
            cv::Point p2 = keypoints.at(body_part1[i + 1]);
            p2.x = obj_pub.camInfo.u + p2.x;
            p2.y = obj_pub.camInfo.v + p2.y;
            cv::line(matrix2, p, p2, cv::Scalar(0, 0, 255), 1);
          }
        }
        // draw legs
        int body_part2[7] = { 11, 10, 9, 1, 12, 13, 14 };
        unsigned int body_part2_size = sizeof(body_part2) / sizeof(*body_part2);
        for (unsigned int i = 0; i < body_part2_size - 1; i++)
        {
          if ((keypoints.at(body_part2[i]).x != 0 || keypoints.at(body_part2[i]).y != 0) &&
              (keypoints.at(body_part2[i + 1]).x != 0 || keypoints.at(body_part2[i + 1]).y != 0))
          {
            cv::Point p = keypoints.at(body_part2[i]);
            p.x = obj_pub.camInfo.u + p.x;
            p.y = obj_pub.camInfo.v + p.y;
            cv::Point p2 = keypoints.at(body_part2[i + 1]);
            p2.x = obj_pub.camInfo.u + p2.x;
            p2.y = obj_pub.camInfo.v + p2.y;
            cv::line(matrix2, p, p2, cv::Scalar(255, 0, 255), 1);
          }
        }
      }
    }

    if (!pedObjs.empty())  // do things only when there is pedestrian
    {
      msgs::PedObjectArray msg_pub;

      msg_pub.header = msg->header;
      msg_pub.header.frame_id = msg->header.frame_id;
      msg_pub.header.stamp = msg->header.stamp;
      msg_pub.objects.assign(pedObjs.begin(), pedObjs.end());
      chatter_pub.publish(msg_pub);

      cv::Rect box;

      // draw each pedestrian's bbox and cross probablity
      for (auto const& obj : msg_pub.objects)
      {
        box.x = obj.camInfo.u;
        box.y = obj.camInfo.v;
        box.width = obj.camInfo.width;
        box.height = obj.camInfo.height;
        cv::rectangle(matrix2, box.tl(), box.br(), CV_RGB(0, 255, 0), 1);
        if (box.y >= 10)
          box.y -= 10;
        else
          box.y = 0;

        std::string probability;
        int p = 100 * obj.crossProbability;
        if (p >= cross_threshold)
        {
          if (show_probability)
            probability = "C(" + std::to_string(p / 100) + "." + std::to_string(p % 100) + ")";
          else
            probability = "C";

          cv::putText(matrix2, probability, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1 /*font size*/, cv::Scalar(0, 50, 255),
                      2, 4, 0);
        }
        else
        {
          if (show_probability)
          {
            if (p >= 10)
              probability = "NC(" + std::to_string(p / 100) + "." + std::to_string(p % 100) + ")";
            else
              probability = "NC(" + std::to_string(p / 100) + ".0" + std::to_string(p % 100) + ")";
          }
          else
            probability = "NC";

          cv::putText(matrix2, probability, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1 /*font size*/,
                      cv::Scalar(100, 220, 0), 2, 4, 0);
        }
      }
      // do resize only when computer cannot support
      // cv::resize(matrix2, matrix2, cv::Size(matrix2.cols / 1, matrix2.rows / 1));

      // make cv::Mat to sensor_msgs::Image
      sensor_msgs::ImageConstPtr msg_pub2 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", matrix2).toImageMsg();

      box_pub.publish(msg_pub2);
    }
#if USE_GLOG
    stop = ros::Time::now();
    total_time += stop - start;
    std::cout << "total time: " << total_time << " sec / loop: " << count << std::endl;
#endif
  }
}

// extract features and pass to random forest model
// return cross probability
float PedestrianEvent::crossing_predict(float bb_x1, float bb_y1, float bb_x2, float bb_y2,
                                        std::vector<cv::Point2f> keypoint, int id, ros::Time time)
{
  try
  {
    if (!keypoint.empty())
    {
      std::vector<float> keypoints_x;
      std::vector<float> keypoints_y;

      // Get body we need
      int body_part[13] = { 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14 };
      int body_part_size = sizeof(body_part) / sizeof(*body_part);
      for (int i = 0; i < body_part_size; i++)
      {
        keypoints_x.insert(keypoints_x.end(), keypoint[body_part[i]].x);
        keypoints_y.insert(keypoints_y.end(), keypoint[body_part[i]].y);
      }
      // Calculate the features
      int keypoints_num = body_part_size;
      std::vector<float> feature;

      // Add bbox to feature vector
      float bbox[] = { bb_x1, bb_y1, bb_x2, bb_y2 };
      feature.insert(feature.end(), bbox, bbox + sizeof(bbox) / sizeof(bbox[0]));

      // Calculate x_distance, y_distance, distance, angle
      for (int m = 0; m < keypoints_num; m++)
      {
        for (int n = m + 1; n < keypoints_num; n++)
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

      for (int m = 0; m < keypoints_num; m++)
      {
        for (int n = m + 1; n < keypoints_num; n++)
        {
          for (int k = n + 1; k < keypoints_num; k++)
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

#if USE_GLOG
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
    else
    {
      cv::Mat feature_mat = cv::Mat(1, 4, CV_32F, { bb_x1, bb_y1, bb_x2, bb_y2 });
      // Predict
      float predict_result = predict_rf(feature_mat);

      return predict_result;
    }
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
      angle[2] = M_PI;
    else if (std::max(a, std::max(b, c)) == b)
      angle[0] = M_PI;
    else
      angle[1] = M_PI;
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

#if USE_GLOG
  std::cout << "prediction: " << p << votes.size() << std::endl;
  std::cout << votes.at<int>(0, 0) << " " << votes.at<int>(0, 1) << std::endl;
  std::cout << votes.at<int>(1, 0) << " " << votes.at<int>(1, 1) << std::endl;
#endif

  return p;
}
// use random forest model to predict cross probability
// return cross probability
float PedestrianEvent::predict_rf(cv::Mat input_data)
{
  float p = rf->predict(input_data);

#if USE_GLOG
  std::cout << "prediction: " << p << std::endl;
#endif

  return p;
}

bool PedestrianEvent::too_far(const msgs::BoxPoint box_point)
{
  if ((box_point.p0.x + box_point.p6.x) / 2 > max_distance)
    return true;
  else
    return false;
}

void PedestrianEvent::pedestrian_event()
{
  // AsyncSpinner reference:
  //  https://gist.github.com/bgromov/45ebeced9e8067d9f13cceececc00d5b#file-test_spinner-cpp-L63

  // custom callback queue
  ros::CallbackQueue queue;

  // This node handle uses global callback queue
  ros::NodeHandle n;
  // and this one uses custom queue
  ros::NodeHandle hb_n;
  // Set custom callback queue
  hb_n.setCallbackQueue(&queue);

  ros::Subscriber sub;
  ros::Subscriber sub2;
  if (input_source == 0)
  {
    sub = n.subscribe("/CamObjFrontCenter", 1, &PedestrianEvent::chatter_callback,
                      this);  // /CamObjFrontCenter is sub topic
    sub2 = hb_n.subscribe("/cam/F_center", 1, &PedestrianEvent::cache_image_callback,
                          this);  // /cam/F_center is sub topic
  }
  else if (input_source == 1)
  {
    sub = n.subscribe("/CamObjFrontLeft", 1, &PedestrianEvent::chatter_callback,
                      this);  // /CamObjFrontLeft is sub topic
    sub2 = hb_n.subscribe("/cam/F_left", 1, &PedestrianEvent::cache_image_callback, this);  // /cam/F_left is sub topic
  }
  else if (input_source == 2)
  {
    sub = n.subscribe("/CamObjFrontRight", 1, &PedestrianEvent::chatter_callback,
                      this);  // /CamObjFrontRight is sub topic
    sub2 = hb_n.subscribe("/cam/F_right", 1, &PedestrianEvent::cache_image_callback,
                          this);  // /cam/F_right is sub topic
  }
  else  // input_source == 3
  {
    sub = n.subscribe("/PathPredictionOutput/camera", 1, &PedestrianEvent::chatter_callback,
                      this);  // /CamObjFrontRight is sub topic
    sub2 = hb_n.subscribe("/cam/F_center", 1, &PedestrianEvent::cache_image_callback,
                          this);  // /cam/F_right is sub topic
  }

  // Create AsyncSpinner, run it on all available cores and make it process custom callback queue
  g_spinner.reset(new ros::AsyncSpinner(0, &queue));

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
      queue.clear();
      // Start the spinner
      g_spinner->start();
      ROS_INFO("Spinner enabled");
      // Reset trigger
      g_trigger = false;
    }

    // Process messages on global callback queue
    ros::spinOnce();
    loop_rate.sleep();
  }
  // Release AsyncSpinner object
  g_spinner.reset();

  // Wait for ROS threads to terminate
  ros::waitForShutdown();
}

// return 25 keypoints detected by openpose
std::vector<cv::Point2f> PedestrianEvent::get_openpose_keypoint(cv::Mat input_image)
{
#if USE_GLOG
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
      op::opLog("\nKeypoints:");
      // Accesing each element of the keypoints
      const auto& poseKeypoints = datumProcessed->at(0)->poseKeypoints;
      op::opLog("Person pose keypoints:");
      for (auto person = 0; person < poseKeypoints.getSize(0); person++)
      {
        op::opLog("Person " + std::to_string(person) + " (x, y, score):");
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
          op::opLog(valueToPrint);
        }
      }
    }
  }

#if USE_GLOG
  std::cout << "Openpose time cost: " << ros::Time::now() - timer << std::endl;
#endif

  for (int i = 0; i < 25; i++)
    points.push_back(cv::Point2f(0.0, 0.0));

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
    op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
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
}
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

  //std::string protoFile = PED_MODEL_DIR + std::string("/body_25/pose_deploy.prototxt");
  //std::string weightsFile = PED_MODEL_DIR + std::string("/body_25/pose_iter_584000.caffemodel");

  std::string bash = PED_MODEL_DIR + std::string("/../download_models.sh");
  system(bash.c_str());
  //pe.net_openpose = cv::dnn::readNetFromCaffe(protoFile, weightsFile);
  pe.rf = cv::ml::StatModel::load<cv::ml::RTrees>(PED_MODEL_DIR + std::string("/rf.yml"));
  pe.rf_pose = cv::ml::StatModel::load<cv::ml::RTrees>(PED_MODEL_DIR + std::string("/rf_3frames_normalization.yml"));

  ros::NodeHandle nh;
  pe.chatter_pub =
      nh.advertise<msgs::PedObjectArray>("/PedCross/Pedestrians", 1);  // /PedCross/Pedestrians is pub topic
  ros::NodeHandle nh2;
  pe.box_pub = nh2.advertise<sensor_msgs::Image&>("/PedCross/DrawBBox", 1);  // /PedCross/DrawBBox is pub topic
  ros::NodeHandle nh3;
  pe.pose_pub = nh3.advertise<sensor_msgs::Image&>("/PedCross/CroppedBox", 1);  // /PedCross/CroppedBox is pub topic

  // Get parameters from ROS
  nh.getParam("/show_probability", pe.show_probability);
  nh.getParam("/input_source", pe.input_source);
  nh.getParam("/max_distance", pe.max_distance);

  pe.imageCache = boost::circular_buffer<std::pair<ros::Time, cv::Mat>>(pe.buffer_size);
  pe.buffer.initial();

  pe.openPoseROS();

  stop = ros::Time::now();
  std::cout << "PedCross started. Init time: " << stop - start << " sec" << std::endl;
  pe.count = 0;
  pe.run();
  return 0;
}
