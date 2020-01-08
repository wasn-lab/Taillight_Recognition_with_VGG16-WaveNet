/*
   CREATER: ICL U300
   DATE: Aug, 2019
*/
#include "openroadnet.h"
#include "main.h"
#include "nppi_geometry_transforms.h"
#include "npp.h"

openroadnet opn;

using namespace npp_wrapper;

npp npp_main;

void callback_cam(const sensor_msgs::Image::ConstPtr& msg)  // const sensor_msgs::Image::ConstPtr &msg
// void callback_cam(sensor_msgs::CompressedImage compressImg) //const sensor_msgs::Image::ConstPtr &msg
{
  // cv::imdecode(cv::Mat(compressImg.data),1).copyTo(mat60);

  cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

  mat60 = cv_ptr->image;

  cv::Size dsize = cv::Size(1920, 1208);
  mat60_clone = cv::Mat(dsize, CV_8UC3);
  npp_main.resize3(mat60, mat60_clone, 1208, 1920);

  // mat60_clone = mat60.clone();

  // Npp8u* tmpppp;

  // int dummy;
  // tmpppp = nppiMalloc_8u_C3(1920, 1208, &dummy);
  // cudaMemcpy(tmpppp, mat60_clone.data, mat60_clone.cols*mat60_clone.rows*3, cudaMemcpyHostToDevice);

  OpenRoadNet_output_pub_main = opn.run(mat60_clone);
  OpenRoadNet_pub.publish(OpenRoadNet_output_pub_main);
  // display_result(mat60_clone, opn.dis_free_);

  if (display_flag)
  {
    opn.display_result(mat60_clone, opn.dis_free_);
    // msg_img = cv_bridge::CvImage(std_msgs::Header(), "rgb8", mat60_clone).toImageMsg();
    // pub.publish(msg_img);

    cv_bridge::CvImage cv_image;
    cv_image.image = mat60_clone;
    cv_image.encoding = "bgr8";
    sensor_msgs::Image ros_image;
    cv_image.toImageMsg(ros_image);

    pub.publish(ros_image);
  }

  // cv::imwrite("_tsttt"+ std::to_string(count) + ".jpg", mat60_clone);
  // count += 1;

  // nppiFree(tmpppp);
}

void openroadnet_publish()
{
  ros::Rate loop1(30);
  while (ros::ok())
  {
    OpenRoadNet_pub.publish(OpenRoadNet_output_pub_main);
    loop1.sleep();
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "openroadnet");
  ros::NodeHandle nh;

  ros::Subscriber cam60 = nh.subscribe("/gmsl_camera/1", 4, callback_cam);
  // ros::Subscriber cam60 = nh.subscribe("gmsl_camera/port_a/cam_1/image_raw/compressed", 4, callback_cam);

  OpenRoadNet_pub = nh.advertise<msgs::FreeSpaceResult>("/OpenRoadNet", 8);

  // image_transport::ImageTransport it(nh);

  ros::param::get(ros::this_node::getName() + "/display", display_flag);
  if (display_flag)
    pub = nh.advertise<sensor_msgs::Image>("/OpenRoadNet_img", 1);

  std::string pkg_path = ros::package::getPath("openroadnet");

  // pkg_path_pb = pkg_path + "/frozen_inference_graph_4k.pb";
  pkg_path_pb = pkg_path + "/frozen_inference_graph_x65_5k.pb";

  // pkg_path_json = pkg_path + "/right120.json";
  std::cout << pkg_path_pb << std::endl;

  // const char *pb_path_c = pkg_path_pb.c_str();

  // TF initialize
  opn.init(pkg_path_pb);
  npp_main.init_3(608, 352, 1920, 1208);
  // opn.init(pkg_path_pb, pkg_path_json,2);

  ros::Rate loop(30);
  // std::thread mThread(openroadnet_publish);

  while (ros::ok())
  {
    ros::spinOnce();
    loop.sleep();
  }
  // mThread.join();

  return 0;
}
