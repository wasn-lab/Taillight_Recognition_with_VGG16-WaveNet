#include "ros/ros.h"
#include "std_msgs/Header.h"
#include "std_msgs/String.h"
#include "msgs/Rad.h"
#include "msgs/PointXYZV.h"
#include <cstring>
#include <visualization_msgs/Marker.h>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <netinet/in.h> /* for htons() */

#include <linux/if_packet.h>
#include <linux/if_ether.h> /* for ETH_P_CAN */
#include <linux/can.h>      /* for struct can_frame */

ros::Publisher marker_pub;



void callbackRadFront(const msgs::Rad::ConstPtr &msg)
{
    std_msgs::Header h = msg->radHeader;
    printf("h.stamp.sec: %d\n", h.stamp.sec);
    printf("h.stamp.nsec: %d\n", h.stamp.nsec);
    printf("h.seq: %d\n", h.seq);

/*
    for (int i = 0; i < msg->radPoint.size(); i++)
    {	
	if(msg->radPoint[i].speed<70)
	{
        	ROS_INFO("radPoint(x, y, z, speed)=(%8.4f, %8.4f, %8.4f, %8.4f)", (float)msg->radPoint[i].x, (float)msg->radPoint[i].y, (float)msg->radPoint[i].z, (float)msg->radPoint[i].speed);
	}
        //ROS_INFO("radPoint(x, y, z, speed)=(%8.4f, %8.4f, %8.4f, %8.4f)", (float)msg->radPoint[i].x, (float)msg->radPoint[i].y, (float)msg->radPoint[i].z, (float)msg->radPoint[i].speed);
    }
*/

/*
	float min = 100;
	float x = 0;
	float y = 0;
 	float speed = 0;
    for (int i = 0; i < msg->radPoint.size(); i++)
    {	
	
	if(msg->radPoint[i].speed<70 && msg->radPoint[i].y< 2 && msg->radPoint[i].y>-2)
	{
		if(msg->radPoint[i].x < min)
		{
			min = msg->radPoint[i].x;
			x = msg->radPoint[i].x;
			y = msg->radPoint[i].y;
			speed = msg->radPoint[i].speed;
		}

	}
        //ROS_INFO("radPoint(x, y, z, speed)=(%8.4f, %8.4f, %8.4f, %8.4f)", (float)msg->radPoint[i].x, (float)msg->radPoint[i].y, (float)msg->radPoint[i].z, (float)msg->radPoint[i].speed);
    }
	ROS_INFO("radPoint(x, y, speed)=(%8.4f, %8.4f, %8.4f)", x, y, speed);
*/

//----------------------------------------------------------------------------------------------------------------------------------------------------------
    ros::NodeHandle n;  
    uint32_t shape = visualization_msgs::Marker::SPHERE;
    visualization_msgs::Marker marker;
    // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    marker.header.frame_id = "/base_link";
    marker.header.stamp = ros::Time::now();

    // Set the namespace and id for this marker.  This serves to create a unique ID
    // Any marker sent with the same namespace and id will overwrite the old one
    marker.ns = "RadarPlotter";
    marker.id = 0;

    // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    marker.type = shape;

    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.action = visualization_msgs::Marker::ADD;

    // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
    //marker.pose.position.x = (float)msg->radPoint[i].x;
    //marker.pose.position.y = (float)msg->radPoint[i].y;
    marker.pose.position.z = 0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 0.5;

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;

    // Set the color -- be sure to set alpha to something non-zero!
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;

    marker.lifetime = ros::Duration();

    // Publish the marker
    //marker_pub.publish(marker);
    while (marker_pub.getNumSubscribers() < 1)
    {
      if (!ros::ok())
      {
	break;
      }
      ROS_WARN_ONCE("Please create a subscriber to the marker");
      sleep(1);
    }
    float min = 100;
    float x = 1000;
    float y = 1000;
    float speed = 0;
    for (int i = 0; i < msg->radPoint.size(); i++)
    {	
 	
	if(msg->radPoint[i].speed<70 && msg->radPoint[i].speed>0.1 )
        {
	    if((msg->radPoint[i].x) < min)
            {
		min = (msg->radPoint[i].x);
	        x = (msg->radPoint[i].x);
	        y = msg->radPoint[i].y;
	        speed = msg->radPoint[i].speed;
	    }
	     
        }
	
    }
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker_pub.publish(marker);
    if(min<100){
        ROS_INFO("radPoint(x, y, speed)=(%8.4f, %8.4f, %8.4f)", x, y, speed);
    }
    
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "RadFrontSub");  
    ros::NodeHandle n;
    ros::Subscriber RadFrontSub = n.subscribe("RadFront", 1, callbackRadFront);
    marker_pub = n.advertise<visualization_msgs::Marker>("RadarPlotter", 1); 
    
    while (ros::ok())
    {
        ros::spinOnce();
    }
    return 0;
}
