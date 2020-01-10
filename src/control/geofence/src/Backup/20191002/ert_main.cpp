
#include <stddef.h>
#include <stdio.h>                     // This ert_main.c example uses printf/fflush 
#include "untitled1.h"                 // Model's header file
#include "rtwtypes.h"
#include <iostream> 
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
static double ABS_BB[800] = {}; //Maximum 100 objects
static double Relative_Speed[100] = {}; 



void LocalizationToVehCallback(const msgs::LocalizationToVeh::ConstPtr& LTVmsg)
{
	Heading = LTVmsg->heading;
	SLAM_x = LTVmsg->x;
	SLAM_y = LTVmsg->y;
}


void chatterCallback1(const msgs::DetectedObjectArray::ConstPtr& msg)
{


	int size = msg->objects.size();
	
	rtObj.rtU.Input3 = size;
	//cout << "Size = " << size  << endl;
			
	for (int i =0 ; i<size ; i++)
	{
		//rtObj.rtU.Input2[0] = -6.5027; 
		//rtObj.rtU.Input2[1] = -92.8962; 
		//cout << " Object_" << i << ": " << endl;
		Relative_Speed[i] = (msg->objects[i].relSpeed);		
		rtObj.rtU.Input2[2*i] = (msg->objects[i].bPoint.p0.x);
		rtObj.rtU.Input2[2*i+1] = (msg->objects[i].bPoint.p0.y);
		rtObj.rtU.Input2[2*i+200] = (msg->objects[i].bPoint.p3.x);
		rtObj.rtU.Input2[2*i+201] = (msg->objects[i].bPoint.p3.y);
		rtObj.rtU.Input2[2*i+400] = (msg->objects[i].bPoint.p4.x);
		rtObj.rtU.Input2[2*i+401] = (msg->objects[i].bPoint.p4.y);
		rtObj.rtU.Input2[2*i+600] = (msg->objects[i].bPoint.p7.x);
		rtObj.rtU.Input2[2*i+601] = (msg->objects[i].bPoint.p7.y);
		//cout << "p0: (" << msg->objects[i].bPoint.p0.x << "," << msg->objects[i].bPoint.p0.y << ")" << endl;
		//cout << "p3: (" << msg->objects[i].bPoint.p3.x << "," << msg->objects[i].bPoint.p3.y << ")" << endl;
		//cout << "p4: (" << msg->objects[i].bPoint.p4.x << "," << msg->objects[i].bPoint.p4.y << ")" << endl;
		//cout << "p7: (" << msg->objects[i].bPoint.p7.x << "," << msg->objects[i].bPoint.p7.y << ")" << endl;			
		//cout << "Relative speed: (" << msg->objects[i].track.relative_velocity.x << "," << msg->objects[i].track.relative_velocity.y << ")" << endl;

		ABS_BB[2*i] = cos(Heading)*rtObj.rtU.Input2[2*i] - sin(Heading)*rtObj.rtU.Input2[2*i+1] + SLAM_x;
		ABS_BB[2*i+1] = sin(Heading)*rtObj.rtU.Input2[2*i] + cos(Heading)*rtObj.rtU.Input2[2*i+1] + SLAM_y;
		ABS_BB[2*i+200] = cos(Heading)*rtObj.rtU.Input2[2*i+200] - sin(Heading)*rtObj.rtU.Input2[2*i+201] + SLAM_x;
		ABS_BB[2*i+201] = sin(Heading)*rtObj.rtU.Input2[2*i+200] + cos(Heading)*rtObj.rtU.Input2[2*i+201] + SLAM_y;
		ABS_BB[2*i+400] = cos(Heading)*rtObj.rtU.Input2[2*i+400] - sin(Heading)*rtObj.rtU.Input2[2*i+401] + SLAM_x;
		ABS_BB[2*i+401] = sin(Heading)*rtObj.rtU.Input2[2*i+400] + cos(Heading)*rtObj.rtU.Input2[2*i+401] + SLAM_y;
		ABS_BB[2*i+600] = cos(Heading)*rtObj.rtU.Input2[2*i+600] - sin(Heading)*rtObj.rtU.Input2[2*i+601] + SLAM_x;
		ABS_BB[2*i+601] = sin(Heading)*rtObj.rtU.Input2[2*i+600] + cos(Heading)*rtObj.rtU.Input2[2*i+601] + SLAM_y;
		for (int i =0 ; i<400 ; i++)
		{
			rtObj.rtU.Input2[i] = ABS_BB[i];
		}
		//cout << "abs_p0: (" << msg->objects[i].bPoint.p0.x << "," << msg->objects[i].bPoint.p0.y << ")" << endl;
		//cout << "abs_p3: (" << msg->objects[i].bPoint.p3.x << "," << msg->objects[i].bPoint.p3.y << ")" << endl;
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

	rtObj.rtU.Input[0] = msg->XP1_0;
	rtObj.rtU.Input[1] = msg->XP1_1; 
	rtObj.rtU.Input[2] = msg->XP1_2;
	rtObj.rtU.Input[3] = msg->XP1_3;
	rtObj.rtU.Input[4] = msg->XP1_4; 
	rtObj.rtU.Input[5] = msg->XP1_5;
	rtObj.rtU.Input[6] = msg->XP2_0; 
	rtObj.rtU.Input[7] = msg->XP2_1;  
	rtObj.rtU.Input[8] = msg->XP2_2;   
	rtObj.rtU.Input[9] = msg->XP2_3; 
	rtObj.rtU.Input[10] = msg->XP2_4; 
	rtObj.rtU.Input[11] = msg->XP2_5;
	rtObj.rtU.Input1[0] = msg->YP1_0;
	rtObj.rtU.Input1[1] = msg->YP1_1; 
	rtObj.rtU.Input1[2] = msg->YP1_2;
	rtObj.rtU.Input1[3] = msg->YP1_3;
	rtObj.rtU.Input1[4] = msg->YP1_4; 
	rtObj.rtU.Input1[5] = msg->YP1_5;
	rtObj.rtU.Input1[6] = msg->YP2_0; 
	rtObj.rtU.Input1[7] = msg->YP2_1;  
	rtObj.rtU.Input1[8] = msg->YP2_2;   
	rtObj.rtU.Input1[9] = msg->YP2_3; 
	rtObj.rtU.Input1[10] = msg->YP2_4; 
	rtObj.rtU.Input1[11] = msg->YP2_5;    

	//cout << "X_poly: (" << msg->XP1_0<< "," << msg->XP2_5 << ")" << endl;
	//cout << "Y_poly: (" << msg->YP1_0<< "," << msg->YP2_5 << ")" << endl;

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

	//cout << "Number of objects: "<< rtObj.rtU.Input3<< endl;
  rtObj.step();
	//cout << "Flag: " << rtObj.rtY.Output<< endl;
	int Index = rtObj.rtY.Output2;
	cout << "Distance: " << rtObj.rtY.Output1 << '\t';
	cout <<"Speed:" <<  Relative_Speed[Index-1] <<endl;
	//Send CAN

/*
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
 
	frame.data[0] = (short int)(rtObj.rtY.Output1*100);
	frame.data[1] = (short int)(rtObj.rtY.Output1*100)>>8;
	frame.data[2] = (short int)(Relative_Speed[Index-1]);
	frame.data[3] = (short int)(Relative_Speed[Index-1])>>8;

	nbytes = write(s, &frame, sizeof(struct can_frame));

	//cout << "CAN_ID = " <<  frame.can_id << endl;

	close(s);
*/
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
  ros::init(argc, argv, "Geofence");
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

