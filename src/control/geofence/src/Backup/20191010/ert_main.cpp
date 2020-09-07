
#include <stddef.h>
#include <stdio.h>                     // This ert_main.c example uses printf/fflush 
#include <iomanip> 
#include <iostream> 
#include "Geofence.h"                 // Model's header file
#include "rtwtypes.h"
#include "std_msgs/Header.h"
#include "msgs/BoxPoint.h"
#include "msgs/DynamicPath.h"
#include "msgs/DetectedObject.h"
#include "msgs/DetectedObjectArray.h"
#include "msgs/PathPrediction.h"
#include "msgs/PointXY.h"
#include "msgs/PointXYZ.h"
#include "msgs/PointXYZV.h"
#include "msgs/TrackInfo.h"
#include "msgs/LocalizationToVeh.h"
#include <cstdlib>
#include "ros/ros.h"
#include <math.h>

//For CAN BUS
#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <typeinfo>
#define CAN_DLC 8;
#define CAN_INTERFACE_NAME "can1"
#define CAN_ID  0x590

  
using namespace std;
static untitled1ModelClass rtObj;      // Instance of model class
static double Heading, SLAM_x, SLAM_y;
static double ABS_BB[1000] = {}; //Maximum 100 objects
 



void LocalizationToVehCallback(const msgs::LocalizationToVeh::ConstPtr& LTVmsg)
{
	Heading = LTVmsg->heading;
	SLAM_x = LTVmsg->x;
	SLAM_y = LTVmsg->y;
}


void chatterCallback1(const msgs::DetectedObjectArray::ConstPtr& msg)
{


	int size = msg->objects.size();
	
	rtObj.rtU.OB_num = size;
	//cout << "Size = " << size  << endl;
			
	for (int i =0 ; i<size ; i++)
	{

		rtObj.rtU.BoundingBox[i] = (msg->objects[i].bPoint.p0.x);
		rtObj.rtU.BoundingBox[i+400] = (msg->objects[i].bPoint.p0.y);
		rtObj.rtU.BoundingBox[i+100] = (msg->objects[i].bPoint.p3.x);
		rtObj.rtU.BoundingBox[i+500] = (msg->objects[i].bPoint.p3.y);
		rtObj.rtU.BoundingBox[i+200] = (msg->objects[i].bPoint.p4.x);
		rtObj.rtU.BoundingBox[i+600] = (msg->objects[i].bPoint.p4.y);
		rtObj.rtU.BoundingBox[i+300] = (msg->objects[i].bPoint.p7.x);
		rtObj.rtU.BoundingBox[i+700] = (msg->objects[i].bPoint.p7.y);
		rtObj.rtU.BoundingBox[i+800] = (msg->objects[i].relSpeed);	
		//cout << "p0: (" << msg->objects[i].bPoint.p0.x << "," << msg->objects[i].bPoint.p0.y << ")" << endl;
		//cout << "p3: (" << msg->objects[i].bPoint.p3.x << "," << msg->objects[i].bPoint.p3.y << ")" << endl;
		//cout << "p4: (" << msg->objects[i].bPoint.p4.x << "," << msg->objects[i].bPoint.p4.y << ")" << endl;
		//cout << "p7: (" << msg->objects[i].bPoint.p7.x << "," << msg->objects[i].bPoint.p7.y << ")" << endl;			
		//cout << "Relative speed: (" << msg->objects[i].track.relative_velocity.x << "," << msg->objects[i].track.relative_velocity.y << ")" << endl;

		ABS_BB[i] = cos(Heading)*rtObj.rtU.BoundingBox[i] - sin(Heading)*rtObj.rtU.BoundingBox[i+400] + SLAM_x;
		ABS_BB[i+400] = sin(Heading)*rtObj.rtU.BoundingBox[i] + cos(Heading)*rtObj.rtU.BoundingBox[i+400] + SLAM_y;
		ABS_BB[i+100] = cos(Heading)*rtObj.rtU.BoundingBox[i+100] - sin(Heading)*rtObj.rtU.BoundingBox[i+500] + SLAM_x;
		ABS_BB[i+500] = sin(Heading)*rtObj.rtU.BoundingBox[i+100] + cos(Heading)*rtObj.rtU.BoundingBox[i+500] + SLAM_y;
		ABS_BB[i+200] = cos(Heading)*rtObj.rtU.BoundingBox[i+200] - sin(Heading)*rtObj.rtU.BoundingBox[i+600] + SLAM_x;
		ABS_BB[i+600] = sin(Heading)*rtObj.rtU.BoundingBox[i+200] + cos(Heading)*rtObj.rtU.BoundingBox[i+600] + SLAM_y;
		ABS_BB[i+300] = cos(Heading)*rtObj.rtU.BoundingBox[i+300] - sin(Heading)*rtObj.rtU.BoundingBox[i+700] + SLAM_x;
		ABS_BB[i+700] = sin(Heading)*rtObj.rtU.BoundingBox[i+300] + cos(Heading)*rtObj.rtU.BoundingBox[i+700] + SLAM_y;
		for (int i =0 ; i<800 ; i++)
		{
			rtObj.rtU.BoundingBox[i] = ABS_BB[i];
		}
		//cout << "abs_p0: (" << msg->objects[i].bPoint.p0.x << "," << msg->objects[i].bPoint.p0.y << ")" << endl;
		//cout << "abs_p4: (" << msg->objects[i].bPoint.p4.x << "," << msg->objects[i].bPoint.p4.y << ")" << endl;
		//cout << "abs_p7: (" << msg->objects[i].bPoint.p7.x << "," << msg->objects[i].bPoint.p7.y << ")" << endl;			

		//PP info
		/*if (msg->objects[i].track.is_ready_prediction==1)
		{
			frame.can_id  = can_id_start+(i*NumOfID)+3 ;
			frame.data[0] = (short int)(msg->objects[i].track.forecasts[4].position.x*100);
			frame.data[1] = (short int)(msg->objects[i].track.forecasts[4].position.x*100)>>8;
			frame.data[2] = (short int)(msg->objects[i].track.forecasts[4].position.y*100);
			frame.data[3] = (short int)(msg->objects[i].track.forecasts[4].position.y*100)>>8;
			frame.data[4] = (short int)(msg->objects[i].track.forecasts[9].position.x*100);
			frame.data[5] = (short int)(msg->objects[i].track.forecasts[9].position.x*100)>>8;
			frame.data[6] = (short int)(msg->objects[i].track.forecasts[9].position.y*100);
			frame.data[7] = (short int)(msg->objects[i].track.forecasts[9].position.y*100)>>8;
			nbytes = write(s, &frame, sizeof(struct can_frame));
			cout << "PP.P1: (" << msg->objects[i].track.forecasts[4].position.x << "," << msg->objects[i].track.forecasts[4].position.y << ")" << endl;
			cout << "PP.P2: (" << msg->objects[i].track.forecasts[9].position.x << "," << msg->objects[i].track.forecasts[9].position.y << ")" << endl;
		
			frame.can_id  = can_id_start+(i*NumOfID)+4 ;
			frame.data[0] = (short int)(msg->objects[i].track.forecasts[14].position.x*100);
			frame.data[1] = (short int)(msg->objects[i].track.forecasts[14].position.x*100)>>8;
			frame.data[2] = (short int)(msg->objects[i].track.forecasts[14].position.y*100);
			frame.data[3] = (short int)(msg->objects[i].track.forecasts[14].position.y*100)>>8;
			frame.data[4] = (short int)(msg->objects[i].track.forecasts[19].position.x*100);
			frame.data[5] = (short int)(msg->objects[i].track.forecasts[19].position.x*100)>>8;
			frame.data[6] = (short int)(msg->objects[i].track.forecasts[19].position.y*100);
			frame.data[7] = (short int)(msg->objects[i].track.forecasts[19].position.y*100)>>8;
			nbytes = write(s, &frame, sizeof(struct can_frame));
			cout << "PP.P3: (" << msg->objects[i].track.forecasts[14].position.x << "," << msg->objects[i].track.forecasts[14].position.y << ")" << endl;
			cout << "PP.P4: (" << msg->objects[i].track.forecasts[19].position.x << "," << msg->objects[i].track.forecasts[19].position.y << ")" << endl;
		}*/
					
	}
	

}


void chatterCallback2(const msgs::DynamicPath::ConstPtr& msg)
{

	rtObj.rtU.X_Poly[0] = msg->XP1_0;
	rtObj.rtU.X_Poly[1] = msg->XP1_1; 
	rtObj.rtU.X_Poly[2] = msg->XP1_2;
	rtObj.rtU.X_Poly[3] = msg->XP1_3;
	rtObj.rtU.X_Poly[4] = msg->XP1_4; 
	rtObj.rtU.X_Poly[5] = msg->XP1_5;
	rtObj.rtU.X_Poly[6] = msg->XP2_0; 
	rtObj.rtU.X_Poly[7] = msg->XP2_1;  
	rtObj.rtU.X_Poly[8] = msg->XP2_2;   
	rtObj.rtU.X_Poly[9] = msg->XP2_3; 
	rtObj.rtU.X_Poly[10] = msg->XP2_4; 
	rtObj.rtU.X_Poly[11] = msg->XP2_5;
	rtObj.rtU.Y_Poly[0] = msg->YP1_0;
	rtObj.rtU.Y_Poly[1] = msg->YP1_1; 
	rtObj.rtU.Y_Poly[2] = msg->YP1_2;
	rtObj.rtU.Y_Poly[3] = msg->YP1_3;
	rtObj.rtU.Y_Poly[4] = msg->YP1_4; 
	rtObj.rtU.Y_Poly[5] = msg->YP1_5;
	rtObj.rtU.Y_Poly[6] = msg->YP2_0; 
	rtObj.rtU.Y_Poly[7] = msg->YP2_1;  
	rtObj.rtU.Y_Poly[8] = msg->YP2_2;   
	rtObj.rtU.Y_Poly[9] = msg->YP2_3; 
	rtObj.rtU.Y_Poly[10] = msg->YP2_4; 
	rtObj.rtU.Y_Poly[11] = msg->YP2_5;
	/*    
	for (int i=0;i<12;i++){
		cout << i << ": " <<rtObj.rtU.X_Poly[i] << "," << rtObj.rtU.Y_Poly[i] <<endl;
	}
	*/


}

	
void rt_OneStep(void);
void rt_OneStep(void)
{
  static boolean_T OverrunFlag = false;


  if (OverrunFlag) {
    rtmSetErrorStatus(rtObj.getRTM(), "Overrun");
    return;
  }

  OverrunFlag = true;
  rtObj.step();
	cout << "Distance: " << setw(10)  <<rtObj.rtY.Range << '\t';
	cout << "Speed:" << setw(10) << rtObj.rtY.Obj_Speed <<endl;
	//Send CAN


	int s;
	int nbytes;
	struct sockaddr_can addr;
	struct can_frame frame;
	struct ifreq ifr;
	
	const char *ifname = CAN_INTERFACE_NAME;

	if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
	{
		perror("Error while opening socket");
	}

	strcpy(ifr.ifr_name, ifname);
	ioctl(s, SIOCGIFINDEX, &ifr);

	addr.can_family  = AF_CAN;
	addr.can_ifindex = ifr.ifr_ifindex;

	//printf("%s at index %d\n", ifname, ifr.ifr_ifindex);

	if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0)
	{
		perror("Error in socket bind");
	}
	
	frame.can_dlc = CAN_DLC ;	
	frame.can_id  = CAN_ID ;
 
	frame.data[0] = (short int)(rtObj.rtY.Range*100);
	frame.data[1] = (short int)(rtObj.rtY.Range*100)>>8;
	frame.data[2] = (short int)(rtObj.rtY.Obj_Speed*100);
	frame.data[3] = (short int)(rtObj.rtY.Obj_Speed*100)>>8;

	nbytes = write(s, &frame, sizeof(struct can_frame));

	//cout << "CAN_ID = " <<  frame.can_id << endl;

	close(s);

  OverrunFlag = false;

}


int main(int argc, char **argv)
{
  // Unused arguments
  (void)(argc);
  (void)(argv);

  // Initialize model
  rtObj.initialize();

  // Attach rt_OneStep to a timer or interrupt service routine with
  //  period 0.01 seconds (the model's base sample time) here.  The
  //  call syntax for rt_OneStep is
  //
	
	
  fflush((NULL));
  ros::init(argc, argv, "Geofence_old");
  ros::NodeHandle n,n1,n2;
  ros::Subscriber Geofence_subscriber1 = n.subscribe("PathPredictionOutput/lidar", 1, chatterCallback1);
  ros::Subscriber Geofence_subscriber2 = n1.subscribe("dynamic_path_para", 1, chatterCallback2);
  ros::Subscriber LTV_sub = n2.subscribe("localization_to_veh", 1, LocalizationToVehCallback);
  ros::Rate loop_rate(10);
  while (ros::ok())
  {
  	ros::spinOnce();
  	rt_OneStep();
  	loop_rate.sleep();	
  }
	
  return 0;
}

