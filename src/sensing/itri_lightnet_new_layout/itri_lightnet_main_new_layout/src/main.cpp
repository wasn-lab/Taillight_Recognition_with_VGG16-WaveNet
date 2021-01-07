#include "main.h"

void imageCallback_30deg(const sensor_msgs::ImageConstPtr &msg)
{
    //printf("imageCallback_30deg!\n");
    try
    {
        camera_30deg_flag = 1;

        cv_ptr_30deg = cv_bridge::toCvCopy(msg, "bgr8");
        if (camera_60deg_flag == 1 && camera_30deg_flag == 1)
        {
            DoNet(cv_ptr_30deg, cv_ptr_60deg, &TL_status_info, &TL_color_info);
        }

        msgs::DetectedLightArray lightarray_msg;
        // for(const auto& detect : lightout.dobj)
        {
            msgs::DetectedLight light_msg;
            light_msg.color_light = TL_color_info.color_light; //0=unknown, 1=Red, 2=Yellow, 3=Green
            light_msg.direction = TL_status_info.ros_msg_total;
            light_msg.distance = TL_status_info.distance_light;
            lightarray_msg.lights.push_back(light_msg);
        }
        lightarray_msg.header.frame_id = "base_link";
        //lightarray_msg.header.stamp = source_stamp;
        Traffic_Light_pub.publish(lightarray_msg);
    }
    catch (cv_bridge::Exception &e)
    {
        camera_30deg_flag = 0;
        printf("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void imageCallback_60deg(const sensor_msgs::ImageConstPtr &msg)
{
    camera_60deg_flag = 0;
    //printf("imageCallback_60deg!!\n");
    try
    {
        cv_ptr_60deg = cv_bridge::toCvCopy(msg, "bgr8");
        camera_60deg_flag = 1;
    }
    catch (cv_bridge::Exception &e)
    {
        camera_60deg_flag = 0;
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void msgPublisher()
{
  std_msgs::Empty empty_msg;
  HeartbeatPub.publish(empty_msg);
}

pthread_t thread1;
int main(int argc, char **argv)
{
    initi_all();
    ros::init(argc, argv, "LightNet_ITRI_Campus");
    ros::Time::init();
    //ros::Rate loop_rate(1000000);

    HeartbeatPub = n.advertise<std_msgs::Empty>("LightNet/heartbeat", 1);
#ifdef display_cv_window
    cv::namedWindow("view_30deg");
    cv::namedWindow("view_60deg");
    cv::startWindowThread();
#endif
    ros::NodeHandle nh;
    ros::NodeHandle nw;

    Traffic_Light_pub = nw.advertise<msgs::DetectedLightArray>("LightResultOutput_ITRI_Campus", 30);

    image_transport::ImageTransport it(nh);
#ifdef read_local_bagfile
    image_transport::Subscriber sub_30 = it.subscribe("/cam/front_top_far_30", 1, imageCallback_30deg);
    image_transport::Subscriber sub_60 = it.subscribe("/cam/front_bottom_60", 1, imageCallback_60deg);

#else
    image_transport::Subscriber sub_30 = it.subscribe("/cam/front_top_far_30", 1, imageCallback_30deg);
    image_transport::Subscriber sub_60 = it.subscribe("/cam/front_bottom_60", 1, imageCallback_60deg);
    //image_transport::Subscriber sub = it.subscribe("/cam/F_center", 1, imageCallback);
#endif
    // ros::spin();

    ros::Rate rate(20);
    while (ros::ok())
    {
        msgPublisher();
        ros::spinOnce();
        rate.sleep();
    }

#ifdef display_cv_window
    cv::destroyWindow("view");
#endif
}