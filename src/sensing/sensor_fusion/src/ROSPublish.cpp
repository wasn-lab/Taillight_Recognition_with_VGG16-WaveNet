#include "ROSPublish.h"
#include <iostream>
using namespace std;
#define ROSMSG_INTERVAL 34 //ms
#define ROS_OUTPUT_TOPIC "LaneToDecisionOutput"

std::mutex m;
std::condition_variable cond_;
std::chrono::milliseconds ms(ROSMSG_INTERVAL);
std::queue<msgs::DetectedObjectArray> msg;
msgs::DetectedObjectArray *msg1;

ROSPublish::ROSPublish()
: interval(ms), run(true)
{
	cout <<" init ros publish" <<endl;
	//lanenet_pub = nh.advertise<msgs::DetectedLaneArray>(ROS_OUTPUT_TOPIC, 1000);
 	fusMsg_pub = nh.advertise<msgs::DetectedObjectArray>("SensorFusion", 2);
}

void ROSPublish::stop() {
     run= false;
     
     printf("stop()\n");
}

ROSPublish::~ROSPublish() {

}

void ROSPublish::tickFuntion() {

	while (run) {
        this_thread::sleep_for(std::chrono::milliseconds(interval));
        std::unique_lock<std::mutex>lk(m);
		cond_.wait(lk,[this] {return !msg.empty();}); // {return msg1 != NULL; });
		{
			msgs::DetectedObjectArray t = msg.front();
			//memcpy(&t, &msg1, sizeof(*msg1)+10);
			//cout <<"do you publish ?" << sizeof(*msg1) <<endl;
			//cout << &msg1 <<endl;
			//cout <<t <<endl;
			fusMsg_pub.publish(t); //*msg1);
		}
		lk.unlock();
		cond_.notify_one();

      if(!run)
       printf("leave tickFuntion\n");
    }
}

 
void ROSPublish::PublishcallbackFunction(msgs::DetectedObjectArray &fusion_msg)
{
		std::unique_lock<std::mutex> lk(m);
		//cout<<"************************************"<<endl;
		//cout <<" sensor fusion  " <<  endl;


		//msg1 = &lane_msg;
		//cout << &msg1 << endl; //*msg1 <<endl;
		cout << fusion_msg <<endl;
		if(!fusion_msg.objects.empty())
		{
			while(!msg.empty())
				msg.pop();
			msg.push(fusion_msg); //*msg1);
		}
		//cout <<"***************************************"<<endl;
		lk.unlock();
		cond_.notify_one();

}
