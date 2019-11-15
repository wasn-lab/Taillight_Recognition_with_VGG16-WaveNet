#include "pedestrian_event.h"
// root folder is ./devel/lib/pedestrian_event/
namespace ped
{
void PedestrianEvent::run()
{
  // std::cout<<"hi"<<std::endl;
  pedestrian_event();
}

void PedestrianEvent::cache_image_callback(const sensor_msgs::Image::ConstPtr& msg)
{
  // std::cout << "image";
  imageCache.push_back(*msg);

  cv_bridge::CvImageConstPtr cv_ptr_image;
  cv_ptr_image = cv_bridge::toCvShare(msg, "bgr8");
  cv::Mat mgs_decode = cv_ptr_image->image;
  std::cout << mgs_decode.rows<<" "<<mgs_decode.cols<<std::endl;

  if (mgs_decode.rows != 0 && mgs_decode.cols != 0)
    imageCacheMat.push_back(mgs_decode);

  if (imageCache.size() > buffer_size)
  {
    imageCache.erase(imageCache.begin());
    imageCacheMat.erase(imageCacheMat.begin());
  }
  //  std::cout << imageCache.size()<<std::endl;
}

void PedestrianEvent::chatter_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  if (!imageCache.empty())
  {
    count++;
    ros::Time start, stop;
    start = ros::Time::now();
    std::cout <<"time stamp: " <<msg->header.stamp<<std::endl;

    std::vector<cv::Mat>::const_iterator it_save_Mat = imageCacheMat.begin();
    std::vector<sensor_msgs::Image>::const_iterator it;
    for (it = imageCache.begin(); it != imageCache.end(); ++it)
    {
      sensor_msgs::Image ptr = *it;
      if (ptr.header.stamp - msg->header.stamp <= ros::Duration(0))
      {
        // std::cout <<"GOT CHA !!!!! time: "<< it->header.stamp<<std::endl;
      }
      else
      {
        break;
      }
      it_save_Mat++;
    }

    it = imageCache.begin();
    // std::cout <<"dalay :"<< msg->header.stamp - it->header.stamp <<std::endl;
    if (it_save_Mat == imageCacheMat.end())
      it_save_Mat--;
    std::cout <<"dalayx:"<< it_save_Mat - imageCacheMat.begin() <<std::endl;
    cv::Mat matrix = *it_save_Mat;
    // cv::imwrite( "/home/itri457854/frame.png", matrix );
    std::vector<msgs::PedObject> pedObjs;
    pedObjs.reserve(msg->objects.end() - msg->objects.begin());
    for (std::vector<msgs::DetectedObject>::const_iterator it = msg->objects.begin(); it != msg->objects.end(); ++it)
    {
      msgs::DetectedObject obj = *it;
      if (obj.classId == 1)
      {
        msgs::PedObject obj_pub;
        obj_pub.header = obj.header;
        obj_pub.header.frame_id = obj.header.frame_id;
        obj_pub.header.stamp = obj.header.stamp;
        obj_pub.classId = obj.classId;
        obj_pub.camInfo = obj.camInfo;
        obj_pub.bPoint = obj.bPoint;

         
        obj_pub.camInfo.u*=0.3167;
        obj_pub.camInfo.v*=0.3179;
        obj_pub.camInfo.width*=0.3167; 
        obj_pub.camInfo.height*=0.3179;

        // Avoid index out of bounds
        if (obj_pub.camInfo.u + obj_pub.camInfo.width > matrix.cols)
        {
          obj_pub.camInfo.width = matrix.cols - obj_pub.camInfo.u;
        }
        if (obj_pub.camInfo.v + obj_pub.camInfo.height > matrix.rows)
        {
          obj_pub.camInfo.height = matrix.rows - obj_pub.camInfo.v;
        }

        std::cout <<matrix.cols<<" "<<matrix.rows<<" "<< obj_pub.camInfo.u<<" "<<obj_pub.camInfo.v<<" "<<
           obj_pub.camInfo.u+obj_pub.camInfo.width<<" "<<obj_pub.camInfo.v+obj_pub.camInfo.height <<std::endl;

        cv::Mat cropedImage = matrix(cv::Rect(obj_pub.camInfo.u, obj_pub.camInfo.v, obj_pub.camInfo.width, obj_pub.camInfo.height));
        // cv::imwrite( "/home/itri457854/frame2.png", cropedImage );

        // max pixel of width or height can only be 368
        int max_pixel = 184;
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
          }resize_width_to = max_pixel;
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
          }resize_height_to = max_pixel;
          aspect_ratio = (float)cropedImage.cols / (float)cropedImage.rows;
          resize_width_to = int(aspect_ratio * resize_height_to);
          std::cout << resize_width_to << std::endl;
        }
        cv::resize(cropedImage, cropedImage, cv::Size(resize_width_to, resize_height_to));

        std::vector<cv::Point> keypoints = get_openpose_keypoint(cropedImage);
        std::cout << keypoints.size() << std::endl;
        
        for(unsigned int i=0;i<keypoints.size();i++)
          cv::circle(cropedImage, keypoints.at(i), 2, cv::Scalar(0, 255, 0));

        // std::vector<int> params;
        // params.resize(3, 0);
        // params[0] = CV_IMWRITE_PNG_COMPRESSION;
        // params[1] = 1;
        // cv::imencode(".png", cropedImage, msg_pub3.data, params);
        sensor_msgs::ImageConstPtr msg_pub3 = cv_bridge::CvImage(std_msgs::Header(), "bgr8",cropedImage).toImageMsg();
        pose_pub.publish(msg_pub3);
        obj_pub.crossProbability = load_model(obj.camInfo.u, obj.camInfo.v, obj.camInfo.width, obj.camInfo.height);
        pedObjs.push_back(obj_pub);
      }
    }
    std::cout<<"here"<<std::endl;
    if (!pedObjs.empty())
    {
      msgs::PedObjectArray msg_pub;

      msg_pub.header = msg->header;
      msg_pub.header.frame_id = msg->header.frame_id;
      msg_pub.header.stamp = msg->header.stamp;
      msg_pub.objects.assign(pedObjs.begin(), pedObjs.end());

      chatter_pub.publish(msg_pub);

      // sensor_msgs::Image msg_pub2;
      // msg_pub2 = *it;
      // std::vector<int> params;
      // params.resize(3, 0);
      // params[0] = CV_IMWRITE_PNG_COMPRESSION;
      // params[1] = 1;
      cv::Rect box;
      cv::Mat matrix2;
      matrix.copyTo(matrix2);
      for (std::vector<msgs::PedObject>::const_iterator it = msg_pub.objects.begin(); it != msg_pub.objects.end(); ++it)
      {
        msgs::PedObject obj = *it;
        box.x = obj.camInfo.u;
        box.y = obj.camInfo.v;
        box.width = obj.camInfo.width;
        box.height = obj.camInfo.height;
        cv::rectangle(matrix2, box.tl(), box.br(), CV_RGB(0, 255, 0), 2);
        if (box.y >= 10)
          box.y -= 10;
        else
          box.y = 0;
        cv::putText(matrix2, std::to_string(obj.crossProbability), box.tl(), cv::FONT_HERSHEY_SIMPLEX, 2/*font size*/,
                    cv::Scalar(0, 255, 255), 2, 4, 0);
      }

      cv::resize(matrix2, matrix2, cv::Size(matrix2.cols / 1, matrix2.rows / 1));
      // cv::imshow("image", matrix2);
      // cv::waitKey(10);
      // sensor_msgs::Image msg_img = msg_pub2;
      // cv::imencode(".png", matrix2, msg_img.data, params);
      // msg_img = cv_bridge::CvImage(std_msgs::Header(), "bgr8",matrix2).toImageMsg();
      sensor_msgs::ImageConstPtr msg_pub2 = cv_bridge::CvImage(std_msgs::Header(), "bgr8",matrix2).toImageMsg();

      // std::cout<<"object num: "<<pedObjs.size()<<std::endl;
      box_pub.publish(msg_pub2);
    }
    stop = ros::Time::now();
    total_time += stop - start;
    std::cout << "total time: " << total_time << " sec / loop: " << count << std::endl;
  }
}

float PedestrianEvent::load_model(float u, float v, float w, float h)
{
  cv::Mat array = cv::Mat(1, 4, CV_32F, { u, v, u + w, v + h });
  float p = rf->predict(array);
  // std::cout<<"prediction: "<<p<<std::endl;
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
  ros::Subscriber sub2 =
      hb_n.subscribe("/cam/F_center", 1, &PedestrianEvent::cache_image_callback,
                     this);  // /gmsl_camera/port_a/cam_1/image_raw/compressed is sub topic

  // Create AsyncSpinner, run it on all available cores and make it process custom callback queue
  g_spinner.reset(new ros::AsyncSpinner(0, &queue));

  g_enable = true;
  g_trigger = true;

  // Loop with 100 Hz rate
  ros::Rate loop_rate(100);
  while (ros::ok())
  {
    // Enable state changed
    if (g_trigger)
    {
      if (g_enable)
      {
        // Clear old callback from the queue
        queue.clear();
        // Start the spinner
        g_spinner->start();
        ROS_INFO("Spinner enabled");
      }
      else
      {
        // Stop the spinner
        g_spinner->stop();
        ROS_INFO("Spinner disabled");
      }
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

std::vector<cv::Point> PedestrianEvent::get_openpose_keypoint(cv::Mat input_image)
{
  ros::Time timer = ros::Time::now();
  int nPoints = 15;
  std::vector<cv::Point> points(nPoints);

  // // max pixel of width or height can only be 368
  // int max_pixel = 184;
  // float aspect_ratio = 0.0;
  // int resize_height_to = 0;
  // int resize_width_to = 0;
  std::cout<<"openpose: "<<input_image.cols<<" "<<input_image.rows<<std::endl;
  // if (input_image.cols >= input_image.rows)
  // {  // width larger than height
  //   if (input_image.cols > max_pixel)
  //   {
  //     resize_width_to = max_pixel;
  //   }
  //   else
  //   {
  //     resize_width_to = input_image.cols;
  //   }resize_width_to = max_pixel;
  //   aspect_ratio = (float)input_image.rows / (float)input_image.cols;
  //   resize_height_to = int(aspect_ratio * resize_width_to);
  //   std::cout << resize_height_to << std::endl;
  // }
  // else
  // {  // height larger than width
  //   if (input_image.rows > max_pixel)
  //   {
  //     resize_height_to = max_pixel;
  //   }
  //   else
  //   {
  //     resize_height_to = input_image.rows;
  //   }resize_height_to = max_pixel;
  //   aspect_ratio = (float)input_image.cols / (float)input_image.rows;
  //   resize_width_to = int(aspect_ratio * resize_height_to);
  //   std::cout << resize_width_to << std::endl;
  // }
  std::cout << ros::Time::now() - timer << std::endl;
  timer = ros::Time::now();
  cv::Mat input_Blob = cv::dnn::blobFromImage(input_image, 1.0 / 255, cv::Size(input_image.cols, input_image.rows),
                                              cv::Scalar(0, 0, 0), false, false);
  std::cout << ros::Time::now() - timer << std::endl;
  timer = ros::Time::now();
  net_openpose.setInput(input_Blob);
  std::cout << ros::Time::now() - timer << std::endl;
  timer = ros::Time::now();
  // net_openpose.setPreferableBackend(cv::dnn::DNN_BACKEND_VKCOM);
  // net_openpose.setPreferableTarget(cv::dnn::DNN_TARGET_VULKAN);
  cv::Mat output = net_openpose.forward();
  std::cout << ros::Time::now() - timer <<"size: "<<output.size<< std::endl;
  timer = ros::Time::now();
  for (int n = 0; n < nPoints; n++)
  {
    cv::Mat probMap(output.size[2], output.size[3], CV_32F, output.ptr(0, n));
    cv::resize(probMap, probMap, cv::Size(input_image.cols, input_image.rows));
    cv::Point maxLoc;
    double prob;
    cv::minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
    if(prob>0.005)
      points[n] = maxLoc;
    else
      points[n] = cv::Point(0,0);
    std::cout << points[n] <<" p: "<<prob<< std::endl;
  }
  std::cout << ros::Time::now() - timer << std::endl;
  timer = ros::Time::now();
  return points;
}
}

int main(int argc, char** argv)
{
  ros::Time::init();
  ros::Time start, stop;
  start = ros::Time::now();
  ros::init(argc, argv, "pedestrian_event");
  // caffe::Caffe::set_mode(caffe::Caffe::GPU);
  // caffe::Net<float> lenet("models/mpi/pose_deploy_linevec_faster_4_stages.prototxt",caffe::TEST);
  // lenet.CopyTrainedLayersFrom("models/mpi/pose_iter_160000.caffemodel");

  ped::PedestrianEvent pe;
  pe.rf = cv::ml::StatModel::load<cv::ml::RTrees>(PED_MODEL_DIR+std::string("/rf.yml"));
  std::string protoFile = PED_MODEL_DIR+std::string("/mpi/pose_deploy_linevec_faster_4_stages.prototxt");
  std::string weightsFile = PED_MODEL_DIR+std::string("/mpi/pose_iter_160000.caffemodel");
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
