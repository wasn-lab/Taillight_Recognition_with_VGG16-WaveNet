#include <stdio.h>
#include <sys/timeb.h>
#include <time.h>
#include <queue>
#include <boost/thread/thread.hpp>
#include "Transmission/UdpClientServer.h"
#include "Transmission/CanReceiver.h"
#include "Transmission/RosModule.hpp"


#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include "std_msgs/String.h"


boost::mutex mutex_queue;
boost::mutex mutex_ros;
std::queue<std::string> q;
std::queue<std::string> obuQueue;
std::queue<std::string> vkQueue;
msgs::DetectedObjectArray detObjArray;
msgs::LidLLA gps;
msgs::TaichungVehInfo vehInfo;
json fps_json_ = {{"key",0}};

const static double PI = 3.14;
double data[10] = {0};
const static std::string PLATE = "ITRI-ADV";
const static int FPS_KEY_LEN = 27;
const static std::string keys[] = 
{
       "FPS_LidarAll", 
       "FPS_LidarDetection",
       "FPS_camF_right",
       "FPS_camF_center",
       "FPS_camF_left",
       "FPS_camF_top",
       "FPS_camR_front",
       "FPS_camR_rear",
       "FPS_camL_front",
       "FPS_camL_rear",
       "FPS_camB_top",
       "FPS_CamObjFrontRight",
       "FPS_CamObjFrontCenter",
       "FPS_CamObjFrontLeft",
       "FPS_CamObjFrontTop",
       "FPS_CamObjRightFront",
       "FPS_CamObjRightBack",
       "FPS_CamObjLeftFront",
       "FPS_CamObjLeftBack",
       "FPS_CamObjBackTop",
       "FPS_current_pose",
       "FPS_veh_info",
       "FPS_dynamic_path_para",
       "FPS_Flag_Info01",
       "FPS_Flag_Info02",
       "FPS_Flag_Info03",
       "FPS_V2X_msg",
};


struct pose
{
        double x;
        double y;
        double z;
        double roll;
        double pitch;
        double yaw;
};

pose current_gnss_pose;

char* log_Time ()
{
    struct  tm      *ptm;
    struct  timeb   stTimeb;
    static  char    szTime[24];
 
    ftime(&stTimeb);
    ptm = localtime(&stTimeb.time);
    sprintf(szTime, "%04d-%02d-%02d %02d:%02d:%02d.%03d", ptm->tm_year+1900, ptm->tm_mon+1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec, stTimeb.millitm);
    szTime[23] = 0;
    return szTime;
}

void callback_detObj (const msgs::DetectedObjectArray& input)
{
    mutex_ros.lock ();
    detObjArray = input;
    mutex_ros.unlock ();
}

void callback_gps (const msgs::LidLLA& input)
{
    mutex_ros.lock ();
    gps = input;
    mutex_ros.unlock ();
}

void callback_veh (const msgs::TaichungVehInfo& input)
{
    mutex_ros.lock ();
    vehInfo = input;
    mutex_ros.unlock ();
}

void callback_gnss2local (const geometry_msgs::PoseStamped::ConstPtr& input)
{
        mutex_ros.lock ();
        tf::Quaternion gnss_q(input->pose.orientation.x, input->pose.orientation.y, input->pose.orientation.z,
                              input->pose.orientation.w);
        tf::Matrix3x3 gnss_m(gnss_q);
        
        current_gnss_pose.x = input->pose.position.x;
        current_gnss_pose.y = input->pose.position.y;
        current_gnss_pose.z = input->pose.position.z;
        gnss_m.getRPY(current_gnss_pose.roll, current_gnss_pose.pitch, current_gnss_pose.yaw);
        mutex_ros.unlock ();
}

void callback_fps(const std_msgs::String::ConstPtr& input)
{
  mutex_ros.lock ();
  std::string jsonString = input->data.c_str();
  fps_json_ = json::parse(jsonString);
  mutex_ros.unlock ();
}

std::string get_msg_type(int id) {
    switch (id) {
/*
        case 0x350:
        {
            return "M8.2.adv001";
        }
        break;
*/
        case 0x351:
        {
            return "M8.2.adv002";
        }
        break;
        default:
        break;
    }
    return "";
}

std::string get_jsonmsg_can (const std::string& type, double *data)
{
    std::string time_string = log_Time ();
    json J1;
    J1["type"] = type;
    J1["plate"] = PLATE;
    J1["deviceID"] = "00:00:00:00:00:01";
    J1["dt"] = time_string;
    if (type == "M8.2.adv002"){
        J1["speed"] = data[0];
        J1["front_brake_pressure"] = data[1];
        J1["rear_brake_pressure"] = data[2];
        J1["steering_wheel_angle"] = data[3];
    }
    return J1.dump ();
}

std::string get_jsonmsg_ros (const std::string& type)
{
    std::string time_string = log_Time ();
    json J1;
    J1["type"] = type;
    J1["plate"] = PLATE;
    J1["deviceID"] = "00:00:00:00:00:01";
    J1["dt"] = time_string;
    if (type == "M8.2.adv001") {
        J1["lat"] = gps.lidar_Lat;
        J1["lon"] = gps.lidar_Lon;
        J1["speed"] = -1;
        J1["bearing"] = -1;
        J1["turn_signal"] = -1;
    }else if (type == "M8.2.adv002"){
        J1["speed"] = vehInfo.ego_speed *3.6;
        J1["front_brake_pressure"] = 0;
        J1["rear_brake_pressure"] = 0;
        J1["steering_wheel_angle"] = 0;
    }else if (type == "M8.2.adv003") {
        J1["sw_camera_signal"] = -1;
        J1["sw_lidar_signal"] = -1;
        J1["sw_radar_signal"] = -1;
        J1["slam"] = -1;
    	J1["object_list"];
    	int num = 0;
	for (int i = 0; i < detObjArray.objects.size (); i++)
  	{
	    json J2;
            J2["object_adv_P0_x"] = detObjArray.objects[i].bPoint.p0.x;
            J2["object_adv_P0_y"] = detObjArray.objects[i].bPoint.p0.y;
            J2["object_adv_P1_x"] = detObjArray.objects[i].bPoint.p1.x;
            J2["object_adv_P1_y"] = detObjArray.objects[i].bPoint.p1.y;
            J2["object_adv_P2_x"] = detObjArray.objects[i].bPoint.p2.x;
            J2["object_adv_P2_y"] = detObjArray.objects[i].bPoint.p2.y;
            J2["object_adv_P3_x"] = detObjArray.objects[i].bPoint.p3.x;
            J2["object_adv_P3_y"] = detObjArray.objects[i].bPoint.p3.y;
            J2["object_adv_P4_x"] = detObjArray.objects[i].bPoint.p4.x;
            J2["object_adv_P4_y"] = detObjArray.objects[i].bPoint.p4.y;
            J2["object_adv_P5_x"] = detObjArray.objects[i].bPoint.p5.x;
            J2["object_adv_P5_y"] = detObjArray.objects[i].bPoint.p5.y;
            J2["object_adv_P6_x"] = detObjArray.objects[i].bPoint.p6.x;
            J2["object_adv_P6_y"] = detObjArray.objects[i].bPoint.p6.y;
            J2["object_adv_P7_x"] = detObjArray.objects[i].bPoint.p7.x;
            J2["object_adv_P7_y"] = detObjArray.objects[i].bPoint.p7.y;
            J2["object_adv_lat"] = -1;
            J2["object_adv_lon"] = -1;
            J2["object_adv_x"] = (detObjArray.objects[i].bPoint.p0.x + detObjArray.objects[i].bPoint.p7.x)/2;
            J2["object_adv_y"] = (detObjArray.objects[i].bPoint.p0.y + detObjArray.objects[i].bPoint.p7.y)/2;
            J2["object_type"] = detObjArray.objects[i].classId;
            J2["object_status"] = -1;
            J2["object_length"] = -1;
            J1["object_list"] += J2;
            num++;
        }
        J1["object_count"] = num;
    }
    return J1.dump ();
}

std::string get_jsonmsg_to_obu (const std::string& type)
{
    json J1;
    if (type == "M8.2.adv009") {
        J1["type"] = type;
        J1["lat"] = std::to_string(gps.lidar_Lat);
        J1["lon"] = std::to_string(gps.lidar_Lon);
        J1["speed"] = std::to_string(data[0]);
        J1["bearing"] = std::to_string(current_gnss_pose.yaw * 180/PI);
    }
    return J1.dump ();
}

std::string get_jsonmsg_to_vk_server (const std::string& type)
{
    std::string time_string = log_Time ();
    json J1;
    if (type == "M8.2.VK004") {
        J1["type"] = type;
        J1["deviceid"] = PLATE;
        J1["receivetime"] = time_string ;

        for(int i = 0; i < FPS_KEY_LEN; i ++ )
        {
          std::string key = keys[i];
          float value = fps_json_.value(key, -1);;
          J1[key] = value;
        }
    }
    return J1.dump ();
}

void sendRun (int argc, char ** argv)
{
    UdpClient UDP_SEV_client;
    UdpClient UDP_SEV_OBU;
    UdpClient UDP_SEV_VK_client;

    UDP_SEV_client.initial ("52.69.10.200", 5570);
    UDP_SEV_OBU.initial ("192.168.1.200", 9999);
    UDP_SEV_VK_client.initial("140.96.180.120", 8016);
    
    while (true)
    {
        mutex_queue.lock ();
	while(q.size() != 0)
	{
            int resault =  UDP_SEV_client.send_obj_to_server (q.front());
	    q.pop();
	}        
       
	while(obuQueue.size() != 0)
	{
            int result =  UDP_SEV_OBU.send_obj_to_server (obuQueue.front());
	    obuQueue.pop();
	}        

        while(vkQueue.size() != 0)
	{
            int result =  UDP_SEV_VK_client.send_obj_to_server (vkQueue.front());
	    vkQueue.pop();
	}        
        mutex_queue.unlock ();

	boost::this_thread::sleep (boost::posix_time::microseconds (1000));
    }
}

void receiveCanRun (int argc, char ** argv)
{
    CanReceiver receiver;
    while (true)
    {
        receiver.initial ();
        for(int i = 0; i<1; i++)
        {
       	   
            int msg_id = receiver.receive(data);
            std::string type = get_msg_type(msg_id);
            std::string temp = get_jsonmsg_can(type, data);
            mutex_queue.lock ();
	    q.push(temp);
            mutex_queue.unlock ();
        }
	receiver.closeSocket ();
        boost::this_thread::sleep (boost::posix_time::microseconds (500000));
    }
}

void receiveRosRun (int argc, char ** argv)
{
    RosModuleTraffic::RegisterCallBack (callback_detObj, callback_gps, callback_veh, callback_gnss2local, callback_fps);
    while (ros::ok ())
    {
        mutex_ros.lock ();

            std::string temp_adv001 = get_jsonmsg_ros ("M8.2.adv001");
            mutex_queue.lock ();
            q.push(temp_adv001);
            mutex_queue.unlock ();
/*
            std::string temp_adv002 = get_jsonmsg_ros ("M8.2.adv002");
            mutex_queue.lock ();
            q.push(temp_adv002);
            mutex_queue.unlock ();
*/
            std::string temp_adv003 = get_jsonmsg_ros ("M8.2.adv003");
            mutex_queue.lock ();
            q.push(temp_adv003);
            mutex_queue.unlock ();

            std::string temp_adv009 = get_jsonmsg_to_obu ("M8.2.adv009");
            mutex_queue.lock ();
            obuQueue.push(temp_adv009);
            mutex_queue.unlock ();

            std::string temp_vk004 = get_jsonmsg_to_vk_server ("M8.2.VK004");
            mutex_queue.lock ();
            vkQueue.push(temp_vk004);
            mutex_queue.unlock ();

        mutex_ros.unlock ();
        boost::this_thread::sleep (boost::posix_time::microseconds (500000));
	ros::spinOnce ();
    }
}

int main(int argc, char **argv) {
    RosModuleTraffic::Initial (argc, argv);
    boost::thread TheadCanReceive (receiveCanRun, argc, argv);
    boost::thread TheadRosReceive (receiveRosRun, argc, argv);
    boost::thread TheadSend (sendRun, argc, argv);
    
    while (true)
    {
        boost::this_thread::sleep (boost::posix_time::microseconds (1000000));
    }
    
    return 0;
}
