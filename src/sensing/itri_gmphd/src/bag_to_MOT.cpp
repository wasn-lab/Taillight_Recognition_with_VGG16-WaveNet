#include "bag_to_MOT.h"

namespace ped
{
void BagToMOT::run()
{
  pedestrian_event();
}

void BagToMOT::cache_image_callback(const sensor_msgs::Image::ConstPtr& msg)
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

void BagToMOT::chatter_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  if (!image_cache.empty())  // do if there is image in buffer
  {
    count++;
    ros::Time start, stop;
    start = ros::Time::now();

    // keep original image
    cv::Mat matrix;
    // for painting
    cv::Mat matrix2;
    bool get_timestamp = false;
    ros::Time msgs_timestamp;
    std::vector<msgs::PedObject> pedObjs;
    std::vector<msgs::DetectedObject> alertObjs;
    pedObjs.reserve(msg->objects.end() - msg->objects.begin());

    for (auto const& obj : msg->objects)
    {
      // set msg infomation
      msgs::PedObject obj_pub;
      obj_pub.header = obj.header;
      obj_pub.header.frame_id = obj.header.frame_id;
      obj_pub.header.stamp = obj.header.stamp;
      obj_pub.classId = obj.classId;
      obj_pub.camInfo = obj.camInfo;
      obj_pub.bPoint = obj.bPoint;

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
      }

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
      file << count << ",-1," << obj_pub.camInfo.u << "," << obj_pub.camInfo.v << "," << obj_pub.camInfo.width << ","
           << obj_pub.camInfo.height << "," << obj_pub.camInfo.prob << ",-1,-1,-1\n";
      if (obj_pub.classId == 1)
      {
        file_1 << count << ",-1," << obj_pub.camInfo.u << "," << obj_pub.camInfo.v << "," << obj_pub.camInfo.width
               << "," << obj_pub.camInfo.height << "," << obj_pub.camInfo.prob << ",-1,-1,-1\n";
      }
      if (obj_pub.classId == 2)
      {
        file_2 << count << ",-1," << obj_pub.camInfo.u << "," << obj_pub.camInfo.v << "," << obj_pub.camInfo.width
               << "," << obj_pub.camInfo.height << "," << obj_pub.camInfo.prob << ",-1,-1,-1\n";
      }
      if (obj_pub.classId == 3)
      {
        file_2 << count << ",-1," << obj_pub.camInfo.u << "," << obj_pub.camInfo.v << "," << obj_pub.camInfo.width
               << "," << obj_pub.camInfo.height << "," << obj_pub.camInfo.prob << ",-1,-1,-1\n";
      }
      if (obj_pub.classId == 4)
      {
        file_2 << count << ",-1," << obj_pub.camInfo.u << "," << obj_pub.camInfo.v << "," << obj_pub.camInfo.width
               << "," << obj_pub.camInfo.height << "," << obj_pub.camInfo.prob << ",-1,-1,-1\n";
      }
      if (obj_pub.classId == 5)
      {
        file_2 << count << ",-1," << obj_pub.camInfo.u << "," << obj_pub.camInfo.v << "," << obj_pub.camInfo.width
               << "," << obj_pub.camInfo.height << "," << obj_pub.camInfo.prob << ",-1,-1,-1\n";
      }
      if (obj_pub.classId == 6)
      {
        file_2 << count << ",-1," << obj_pub.camInfo.u << "," << obj_pub.camInfo.v << "," << obj_pub.camInfo.width
               << "," << obj_pub.camInfo.height << "," << obj_pub.camInfo.prob << ",-1,-1,-1\n";
      }
      if (obj_pub.classId == 7)
      {
        file_2 << count << ",-1," << obj_pub.camInfo.u << "," << obj_pub.camInfo.v << "," << obj_pub.camInfo.width
               << "," << obj_pub.camInfo.height << "," << obj_pub.camInfo.prob << ",-1,-1,-1\n";
      }
      if (obj_pub.classId == 8)
      {
        file_2 << count << ",-1," << obj_pub.camInfo.u << "," << obj_pub.camInfo.v << "," << obj_pub.camInfo.width
               << "," << obj_pub.camInfo.height << "," << obj_pub.camInfo.prob << ",-1,-1,-1\n";
      }
    }
    file << count << ",-1,-1,-1,-1,-1,-1,-1,-1,-1\n";
    file_1 << count << ",-1,-1,-1,-1,-1,-1,-1,-1,-1\n";
    file_2 << count << ",-1,-1,-1,-1,-1,-1,-1,-1,-1\n";
    file_3 << count << ",-1,-1,-1,-1,-1,-1,-1,-1,-1\n";
    file_4 << count << ",-1,-1,-1,-1,-1,-1,-1,-1,-1\n";
    file_5 << count << ",-1,-1,-1,-1,-1,-1,-1,-1,-1\n";
    file_6 << count << ",-1,-1,-1,-1,-1,-1,-1,-1,-1\n";
    file_7 << count << ",-1,-1,-1,-1,-1,-1,-1,-1,-1\n";
    file_8 << count << ",-1,-1,-1,-1,-1,-1,-1,-1,-1\n";
    std::stringstream ss_img;
    ss_img << GMPHD_DIR << "/img/";
    int count_tmp = count;
    for (int i = 0; i < 6; i++)
    {
      if (count_tmp == 0)
      {
        ss_img << "0";
      }
      count_tmp /= 10;
    }
    ss_img << count << ".jpg";
    std::cout << ss_img.str() << std::endl;
    std::string img_file_name = ss_img.str();
    imwrite(img_file_name, matrix);

    stop = ros::Time::now();
    total_time += stop - start;

#if PRINT_MESSAGE
    std::cout << "Cost time: " << stop - start << " sec" << std::endl;
    std::cout << "Total time: " << total_time << " sec / loop: " << count << std::endl;
#endif
  }
}

void BagToMOT::pedestrian_event()
{
  // AsyncSpinner reference:
  //  https://gist.github.com/bgromov/45ebeced9e8067d9f13cceececc00d5b#file-test_spinner-cpp-L63

  // custom callback queue
  ros::CallbackQueue queue_1;
  ros::CallbackQueue queue_2;
  ros::CallbackQueue queue_3;
  ros::CallbackQueue queue_4;
  // This node handle uses global callback queue
  ros::NodeHandle nh_sub_1;
  // and this one uses custom queue
  ros::NodeHandle nh_sub_2;
  ros::NodeHandle nh_sub_3;
  ros::NodeHandle nh_sub_4;
  ros::NodeHandle nh_sub_5;
  // Set custom callback queue
  nh_sub_2.setCallbackQueue(&queue_1);
  nh_sub_3.setCallbackQueue(&queue_2);
  nh_sub_4.setCallbackQueue(&queue_3);
  nh_sub_5.setCallbackQueue(&queue_4);

  ros::Subscriber sub_1;  // nh_sub_1
  ros::Subscriber sub_2;
  ros::Subscriber sub_3;  // nh_sub_1
  ros::Subscriber sub_4;
  ros::Subscriber sub_5;
  ros::Subscriber sub_6;
  /**
  front_bottom_60
  left_back_60
  right_back_60
  **/
  sub_1 = nh_sub_1.subscribe("/cam_obj/right_back_60", 1, &BagToMOT::chatter_callback,
                             this);  // /PathPredictionOutput is sub topic
  sub_2 = nh_sub_2.subscribe("/cam/right_back_60", 1, &BagToMOT::cache_image_callback,
                             this);  // /cam/F_right is sub topic

  // Create AsyncSpinner, run it on all available cores and make it process custom callback queue
  g_spinner_1.reset(new ros::AsyncSpinner(0, &queue_1));
  g_spinner_2.reset(new ros::AsyncSpinner(0, &queue_2));
  g_spinner_3.reset(new ros::AsyncSpinner(0, &queue_3));
  g_spinner_4.reset(new ros::AsyncSpinner(0, &queue_4));

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
      // Start the spinner
      g_spinner_1->start();
      g_spinner_2->start();
      g_spinner_3->start();
      g_spinner_4->start();
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
  // Wait for ROS threads to terminate
  ros::waitForShutdown();
}

}  // namespace ped
int main(int argc, char** argv)
{
  ros::Time::init();
  ros::Time start, stop;
  start = ros::Time::now();
  ros::init(argc, argv, "bag_to_MOT");

  ped::BagToMOT pe;

  pe.image_cache = boost::circular_buffer<std::pair<ros::Time, cv::Mat>>(pe.buffer_size);

  stop = ros::Time::now();
  std::cout << "PedCross started. Init time: " << stop - start << " sec" << std::endl;
  pe.count = 0;

  std::stringstream ss;
  ss << GMPHD_DIR;
  // ss << "det.txt";
  std::cout << ss.str() << std::endl;
  std::string fname = ss.str() + "det.txt";
  pe.file.open(fname, std::ios_base::app);
  fname = ss.str() + "det_1.txt";
  pe.file_1.open(fname, std::ios_base::app);
  fname = ss.str() + "det_2.txt";
  pe.file_2.open(fname, std::ios_base::app);
  fname = ss.str() + "det_3.txt";
  pe.file_3.open(fname, std::ios_base::app);
  fname = ss.str() + "det_4.txt";
  pe.file_4.open(fname, std::ios_base::app);
  fname = ss.str() + "det_5.txt";
  pe.file_5.open(fname, std::ios_base::app);
  fname = ss.str() + "det_6.txt";
  pe.file_6.open(fname, std::ios_base::app);
  fname = ss.str() + "det_7.txt";
  pe.file_7.open(fname, std::ios_base::app);
  fname = ss.str() + "det_8.txt";
  pe.file_8.open(fname, std::ios_base::app);
  pe.run();
  pe.file.close();
  pe.file_1.close();
  pe.file_2.close();
  pe.file_3.close();
  pe.file_4.close();
  pe.file_5.close();
  pe.file_6.close();
  pe.file_7.close();
  pe.file_8.close();
  return 0;
}
