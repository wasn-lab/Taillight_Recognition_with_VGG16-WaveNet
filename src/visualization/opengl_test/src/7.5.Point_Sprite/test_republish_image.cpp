#include <ros/ros.h>
// MSG: Image
// #include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>
//
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>


// Debug
#include <iostream>
#include <time_stamp.hpp> // The TIME_STAMP::Time class



using TIME_STAMP::Time;
using TIME_STAMP::Period;


image_transport::Publisher image_pub;

void imageCallback(const sensor_msgs::ImageConstPtr& msg){
    // Re=publish
    image_pub.publish(msg);
    //

}

int main(int argc, char *argv[])
{
    Time t_start(TIME_PARAM::NOW);
    t_start.show();

    Period period_1;

    ros::init(argc, argv, "listener");
    ros::NodeHandle _nh;
    image_transport::ImageTransport _ros_it(_nh);

    // Publish
    image_pub = _ros_it.advertise( "/image_out", 100) ;
    // Subscribe
    image_transport::Subscriber image_sub = _ros_it.subscribe( "/gmsl_camera/port_a/cam_1/image_raw", 100,  imageCallback , ros::VoidPtr(), image_transport::TransportHints("compressed") ) ;


    ros::spin();

    Time t_end(TIME_PARAM::NOW);
    t_end.show();

    Time t_delta = t_end - t_start;
    t_delta.show_sec();

    period_1.stamp();
    period_1.show_sec();

    return 0;
}
