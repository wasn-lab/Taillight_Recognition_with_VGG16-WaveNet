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
  // buffer raw image in cv::Mat with timestamp
  cv_bridge::CvImageConstPtr cv_ptr_image;
  cv_ptr_image = cv_bridge::toCvShare(msg, "bgr8");
  cv::Mat mgs_decode;
  cv_ptr_image->image.copyTo(mgs_decode);

  std::cout << mgs_decode.rows << " " << mgs_decode.cols << std::endl;

  // buffer raw image in msg
  imageCache.emplace_back(msg->header.stamp, mgs_decode);

  // control the size of buffer
  if (imageCache.size() > buffer_size)
  {
    imageCache.pop_front();
  }
}

void PedestrianEvent::chatter_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  if (!imageCache.empty())  // do if there is image in buffer
  {
    count++;
    ros::Time start, stop;
    start = ros::Time::now();
    std::cout << "time stamp: " << msg->header.stamp << " buffer size: " << imageCache.size() << std::endl;

    // compare and get the raw image when object detected
    cv::Mat matrix;
    for (int i = imageCache.size() - 1; i >= 0; i--)
    {
      if (imageCache[i].first <= msg->header.stamp)
      {
        /********  take newer frame if needed
        if(i<imageCache.size()-10)
          i+=9;
          **/
        std::cout << "GOT CHA !!!!! time: " << imageCache[i].first << " , " << msg->header.stamp << std::endl;
        matrix = imageCache[i].second;
        break;
      }
    }

    // for drawing bbox and keypoints
    cv::Mat matrix2;
    matrix.copyTo(matrix2);

    std::vector<msgs::PedObject> pedObjs;
    pedObjs.reserve(msg->objects.end() - msg->objects.begin());
    for (std::vector<msgs::DetectedObject>::const_iterator it = msg->objects.begin(); it != msg->objects.end(); ++it)
    {
      msgs::DetectedObject obj = *it;
      if (obj.classId == 1)  // 1 for people
      {
        // set msg infomation
        msgs::PedObject obj_pub;
        obj_pub.header = obj.header;
        obj_pub.header.frame_id = obj.header.frame_id;
        obj_pub.header.stamp = obj.header.stamp;
        obj_pub.classId = obj.classId;
        obj_pub.camInfo = obj.camInfo;
        obj_pub.bPoint = obj.bPoint;

        // resize from 1920*1208 to 608*384
        obj_pub.camInfo.u *= 0.3167;
        obj_pub.camInfo.v *= 0.3179;
        obj_pub.camInfo.width *= 0.3167;
        obj_pub.camInfo.height *= 0.3179;

        // Avoid index out of bounds
        if (obj_pub.camInfo.u + obj_pub.camInfo.width > matrix.cols)
        {
          obj_pub.camInfo.width = matrix.cols - obj_pub.camInfo.u;
        }
        if (obj_pub.camInfo.v + obj_pub.camInfo.height > matrix.rows)
        {
          obj_pub.camInfo.height = matrix.rows - obj_pub.camInfo.v;
        }

        std::cout << matrix.cols << " " << matrix.rows << " " << obj_pub.camInfo.u << " " << obj_pub.camInfo.v << " "
                  << obj_pub.camInfo.u + obj_pub.camInfo.width << " " << obj_pub.camInfo.v + obj_pub.camInfo.height
                  << std::endl;

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
          std::cout << resize_height_to << std::endl;
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
          std::cout << resize_width_to << std::endl;
        }
        cv::resize(cropedImage, cropedImage, cv::Size(resize_width_to, resize_height_to));

        std::vector<cv::Point> keypoints = get_openpose_keypoint(cropedImage);
        std::cout << keypoints.size() << std::endl;

        // draw keypoints on cropped and whole image
        int body_part[13] = { 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14 };
        for (unsigned int i = 0; i < 13; i++)
        {
          if (keypoints.at(body_part[i]).x != 0 || keypoints.at(body_part[i]).y != 0)
          {
            cv::circle(cropedImage, keypoints.at(body_part[i]), 2, cv::Scalar(0, 255, 0));
            cv::Point p = keypoints.at(body_part[i]);
            p.x = obj_pub.camInfo.u + p.x * obj_pub.camInfo.width / resize_width_to;
            p.y = obj_pub.camInfo.v + p.y * obj_pub.camInfo.width / resize_width_to;
            cv::circle(matrix2, p, 2, cv::Scalar(0, 255, 0), -1);
          }
        }
        // draw hands
        int body_part1[7] = { 4, 3, 2, 1, 5, 6, 7 };
        for (unsigned int i = 0; i < 6; i++)
        {
          if ((keypoints.at(body_part1[i]).x != 0 || keypoints.at(body_part1[i]).y != 0) &&
              (keypoints.at(body_part1[i + 1]).x != 0 || keypoints.at(body_part1[i + 1]).y != 0))
          {
            cv::Point p = keypoints.at(body_part1[i]);
            p.x = obj_pub.camInfo.u + p.x * obj_pub.camInfo.width / resize_width_to;
            p.y = obj_pub.camInfo.v + p.y * obj_pub.camInfo.width / resize_width_to;
            cv::Point p2 = keypoints.at(body_part1[i + 1]);
            p2.x = obj_pub.camInfo.u + p2.x * obj_pub.camInfo.width / resize_width_to;
            p2.y = obj_pub.camInfo.v + p2.y * obj_pub.camInfo.width / resize_width_to;
            cv::line(matrix2, p, p2, cv::Scalar(0, 0, 255), 1);
          }
        }
        // draw legs
        int body_part2[7] = { 11, 10, 9, 1, 12, 13, 14 };
        for (unsigned int i = 0; i < 6; i++)
        {
          if ((keypoints.at(body_part2[i]).x != 0 || keypoints.at(body_part2[i]).y != 0) &&
              (keypoints.at(body_part2[i + 1]).x != 0 || keypoints.at(body_part2[i + 1]).y != 0))
          {
            cv::Point p = keypoints.at(body_part2[i]);
            p.x = obj_pub.camInfo.u + p.x * obj_pub.camInfo.width / resize_width_to;
            p.y = obj_pub.camInfo.v + p.y * obj_pub.camInfo.width / resize_width_to;
            cv::Point p2 = keypoints.at(body_part2[i + 1]);
            p2.x = obj_pub.camInfo.u + p2.x * obj_pub.camInfo.width / resize_width_to;
            p2.y = obj_pub.camInfo.v + p2.y * obj_pub.camInfo.width / resize_width_to;
            cv::line(matrix2, p, p2, cv::Scalar(255, 0, 255), 1);
          }
        }

        sensor_msgs::ImageConstPtr msg_pub3 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cropedImage).toImageMsg();
        pose_pub.publish(msg_pub3);
        bool has_keypoint = false;
        int count_points = 0;
        for (unsigned int i = 0; i < 25; i++)
        {
          if (keypoints.at(i).x != 0 || keypoints.at(i).y != 0)
          {
            count_points++;
            if (count_points >= 3)
              has_keypoint = true;
          }
        }
        if (has_keypoint)
          obj_pub.crossProbability = crossing_predict(obj.camInfo.u, obj.camInfo.v, obj.camInfo.u + obj.camInfo.width,
                                                      obj.camInfo.v + obj.camInfo.height, keypoints);
        else
        {
          std::vector<cv::Point> no_keypoint;
          obj_pub.crossProbability = crossing_predict(obj.camInfo.u, obj.camInfo.v, obj.camInfo.u + obj.camInfo.width,
                                                      obj.camInfo.v + obj.camInfo.height, no_keypoint);
        }
        std::cout << "prob: " << obj_pub.crossProbability << std::endl;
        pedObjs.push_back(obj_pub);
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
      for (std::vector<msgs::PedObject>::const_iterator it = msg_pub.objects.begin(); it != msg_pub.objects.end(); ++it)
      {
        msgs::PedObject obj = *it;
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
        if (p >= 50)
        {
          probability = "C (" + std::to_string(p / 100) + "." + std::to_string(p % 100) + ")";
        }
        else if (p >= 10)
        {
          probability = "NC (" + std::to_string(p / 100) + "." + std::to_string(p % 100) + ")";
        }
        else
        {
          probability = "NC (" + std::to_string(p / 100) + ".0" + std::to_string(p % 100) + ")";
        }
        cv::putText(matrix2, probability, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1 /*font size*/, cv::Scalar(0, 120, 240),
                    2, 4, 0);
      }
      // do resize only when computer cannot support
      // cv::resize(matrix2, matrix2, cv::Size(matrix2.cols / 1, matrix2.rows / 1));

      // make cv::Mat to sensor_msgs::Image
      sensor_msgs::ImageConstPtr msg_pub2 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", matrix2).toImageMsg();

      box_pub.publish(msg_pub2);
    }
    stop = ros::Time::now();
    total_time += stop - start;
    std::cout << "total time: " << total_time << " sec / loop: " << count << std::endl;
  }
}

// extract features and pass to random forest model
// return cross probability
double PedestrianEvent::crossing_predict(double bb_x1, double bb_y1, double bb_x2, double bb_y2,
                                         std::vector<cv::Point> keypoint)
{
  try
  {
    if (!keypoint.empty())
    {
      std::vector<double> keypoints_x;
      std::vector<double> keypoints_y;

      // Get body we need
      int body_part[13] = { 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14 };
      for (int i = 0; i < 13; i++)
      {
        keypoints_x.insert(keypoints_x.end(), keypoint[body_part[i]].x);
        keypoints_y.insert(keypoints_y.end(), keypoint[body_part[i]].y);
      }

      // Caculate the features
      int keypoints_num = 13;
      std::vector<double> feature;

      // Add bbox to feature vector
      double bbox[] = { bb_x1, bb_y1, bb_x2, bb_y2 };
      feature.insert(feature.end(), bbox, bbox + sizeof(bbox) / sizeof(bbox[0]));

      // Caculate x_distance, y_distance, distance, angle
      for (int m = 0; m < keypoints_num; m++)
      {
        for (int n = m + 1; n < keypoints_num; n++)
        {
          double dist_x, dist_y, dist, angle;
          if (keypoints_x[m] != 0.0f && keypoints_y[m] != 0.0f && keypoints_x[n] != 0.0f && keypoints_y[n] != 0.0f)
          {
            dist_x = abs(keypoints_x[m] - keypoints_x[n]);
            dist_y = abs(keypoints_y[m] - keypoints_y[n]);
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
          double input[] = { dist_x, dist_y, dist, angle };
          feature.insert(feature.end(), input, input + sizeof(input) / sizeof(input[0]));
        }
      }

      for (int m = 0; m < keypoints_num; m++)
      {
        for (int n = m + 1; n < keypoints_num; n++)
        {
          for (int k = n + 1; k < keypoints_num; k++)
          {
            double angle[3] = { 0.0f, 0.0f, 0.0f };
            double* angle_ptr;
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
      // Convert vector to array
      static double feature_arr[1174];
      std::copy(feature.begin(), feature.end(), feature_arr);
      // Convert array to Mat
      cv::Mat feature_mat = cv::Mat(1, 1174, CV_32F, feature_arr);
      // Predict
      double predict_result = predict_rf_pose(feature_mat);

      return predict_result;
    }
    else
    {
      cv::Mat feature_mat = cv::Mat(1, 4, CV_32F, { bb_x1, bb_y1, bb_x2, bb_y2 });
      // Predict
      double predict_result = predict_rf(feature_mat);

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
double PedestrianEvent::get_distance2(double x1, double y1, double x2, double y2)
{
  return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

// return degree with line formed by two points and vertical line
double PedestrianEvent::get_angle2(double x1, double y1, double x2, double y2)
{
  return M_PI / 2 - atan2(abs(y1 - y2), abs(x1 - x2));
}

// return 3 inner angles of the triangle formed by three points
double* PedestrianEvent::get_triangle_angle(double x1, double y1, double x2, double y2, double x3, double y3)
{
  double a = get_distance2(x1, y1, x2, y2);
  double b = get_distance2(x2, y2, x3, y3);
  double c = get_distance2(x1, y1, x3, y3);
  double test = (a * a + c * c - b * b) / (2 * a * c);
  static double angle[3] = { 0.0f, 0.0f, 0.0f };
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
double PedestrianEvent::predict_rf_pose(cv::Mat input_data)
{
  cv::Mat votes;
  rf_pose->getVotes(input_data, votes, 0);
  double positive = votes.at<int>(1, 1);
  double negative = votes.at<int>(1, 0);
  double p = positive / (negative + positive);
  std::cout << "prediction: " << p << votes.size() << std::endl;
  std::cout << votes.at<int>(0, 0) << " " << votes.at<int>(0, 1) << std::endl;
  std::cout << votes.at<int>(1, 0) << " " << votes.at<int>(1, 1) << std::endl;
  return p;
}
// use random forest model to predict cross probability
// return cross probability
double PedestrianEvent::predict_rf(cv::Mat input_data)
{
  double p = rf->predict(input_data);
  std::cout << "prediction: " << p << std::endl;
  return p;
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

  ros::Subscriber sub =
      n.subscribe("/CamObjFrontCenter", 1, &PedestrianEvent::chatter_callback, this);  // CamObjFrontCenter is sub topic
  ros::Subscriber sub2 = hb_n.subscribe("/cam/F_center", 1, &PedestrianEvent::cache_image_callback,
                                        this);  // /gmsl_camera/port_a/cam_1/image_raw/compressed is sub topic

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
std::vector<cv::Point> PedestrianEvent::get_openpose_keypoint(cv::Mat input_image)
{
  ros::Time timer = ros::Time::now();
  int nPoints = 25;
  std::vector<cv::Point> points(nPoints);

  std::cout << "openpose: " << input_image.cols << " " << input_image.rows << std::endl;

  cv::Mat input_Blob = cv::dnn::blobFromImage(input_image, 1.0 / 255, cv::Size(input_image.cols, input_image.rows),
                                              cv::Scalar(0, 0, 0), false, false);

  net_openpose.setInput(input_Blob);

  cv::Mat output = net_openpose.forward();
  std::cout << ros::Time::now() - timer << "size: " << output.size << std::endl;
  timer = ros::Time::now();
  for (int n = 0; n < nPoints; n++)
  {
    cv::Mat probMap(output.size[2], output.size[3], CV_32F, output.ptr(0, n));
    cv::resize(probMap, probMap, cv::Size(input_image.cols, input_image.rows));
    cv::Point maxLoc;
    double prob;
    cv::minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
    if (prob > 0.005)
      points[n] = maxLoc;
    else
      points[n] = cv::Point(0, 0);
    std::cout << points[n] << " p: " << prob << std::endl;
  }
  std::cout << ros::Time::now() - timer << std::endl;
  timer = ros::Time::now();
  return points;
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
  // caffe::Caffe::set_mode(caffe::Caffe::GPU);
  // caffe::Net<float> lenet("models/mpi/pose_deploy_linevec_faster_4_stages.prototxt",caffe::TEST);
  // lenet.CopyTrainedLayersFrom("models/mpi/pose_iter_160000.caffemodel");

  ped::PedestrianEvent pe;
  pe.rf = cv::ml::StatModel::load<cv::ml::RTrees>(PED_MODEL_DIR + std::string("/rf.yml"));
  pe.rf_pose = cv::ml::StatModel::load<cv::ml::RTrees>(PED_MODEL_DIR + std::string("/rf_1frame.yml"));
  std::string protoFile = PED_MODEL_DIR + std::string("/body_25/pose_deploy.prototxt");
  std::string weightsFile = PED_MODEL_DIR + std::string("/body_25/pose_iter_584000.caffemodel");
  pe.net_openpose = cv::dnn::readNetFromCaffe(protoFile, weightsFile);

  ros::NodeHandle nh;
  pe.chatter_pub = nh.advertise<msgs::PedObjectArray>("/PedestrianIntention", 1);  // PedestrianIntention is pub topic
  ros::NodeHandle nh2;
  pe.box_pub = nh2.advertise<sensor_msgs::Image&>("/DrawBBox", 1);  // DrawBBox is pub topic
  ros::NodeHandle nh3;
  pe.pose_pub = nh3.advertise<sensor_msgs::Image&>("/OpenPoseBox", 1);  // OpenPoseBox is pub topic

  stop = ros::Time::now();
  std::cout << "init time: " << stop - start << " sec" << std::endl;
  pe.count = 0;
  pe.run();
  return 0;
}
