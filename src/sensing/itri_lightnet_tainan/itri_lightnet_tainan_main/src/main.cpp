#include "main.h"

void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        DoNet(cv_ptr, &TL_status_info, &TL_color_info);

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
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

pthread_t thread1;
int main(int argc, char **argv)
{
    initi_all();

    ros::init(argc, argv, "LightNet");
#ifdef display_cv_window
    cv::namedWindow("view");
    cv::startWindowThread();
#endif
    ros::NodeHandle nh;
    ros::NodeHandle nw;

    Traffic_Light_pub = nw.advertise<msgs::DetectedLightArray>("LightResultOutput", 30);

    image_transport::ImageTransport it(nh);
#ifdef read_local_bagfile    
    image_transport::Subscriber sub = it.subscribe("camera/image_raw", 1, imageCallback);
#else
    image_transport::Subscriber sub = it.subscribe("/cam/F_center", 1, imageCallback);
#endif

    ros::spin();

#ifdef display_cv_window
    cv::destroyWindow("view");
#endif
}