#include <cstdio>
#include <sys/timeb.h>
#include <ctime>
#include <queue>
#include <boost/thread/thread.hpp>
#include "Transmission/UdpClientServer.h"
#include "Transmission/CanReceiver.h"
#include "Transmission/RosModule.h"
#include "Transmission/TCPClient.h"
#include "Transmission/TcpServer.h"

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include "std_msgs/String.h"
#include "Transmission/MqttClient.h"
#include <chrono>

bool g_flag_show_udp_send = true;
bool g_event_queue_switch = true;

// VK APIs backend
const std::string TCP_VK_SRV_ADRR = "60.250.196.127";
const int TCP_VK_SRV_PORT = 55553;

const std::string UDP_VK_SRV_ADRR = "60.250.196.127";
const int UDP_VK_SRV_PORT = 55554;

// aws backend
const std::string UDP_AWS_SRV_ADRR = "52.69.10.200";
const int UDP_AWS_SRV_PORT = 5570;

// OBU
const std::string UDP_OBU_ADRR = "192.168.1.200";
const int UDP_OBU_PORT = 9999;

// TCP Server on ADV
const std::string TCP_ADV_SRV_ADRR = "192.168.1.6";
const int TCP_ADV_SRV_PORT = 8765;

const std::string UDP_ADV_SRV_ADRR = "192.168.1.6";
const int UDP_ADV_SRV_PORT = 8766;

// obu traffic signal
const std::string TOPIC_TRAFFIC = "/traffic";
// Server status
const std::string TOPIC_SERCER_STATUS = "/backend/connected";
// reserve bus
const std::string TOPIC_RESERVE = "/reserve/request";
// route 
const std::string TOPIC_ROUTE = "/reserve/route";


// wait reserve result: 300ms.
const int REVERSE_SLEEP_TIME_MICROSECONDS = 300 * 1000;
//reserve waiting timeout: 3 seconds
const int RESERVE_WAITING_TIMEOUT = 3 * 1000 * 1000;
// UDP server udpate from queues freq 100ms
const int UDP_SERVER_UPDATE_MICROSECONDS = 100 * 1000;
// ROS update time: 100ms
const int ROS_UPDATE_MICROSECONDS = 100 * 1000;
// server status update time: 10 sec
//const int SERVER_STATUS_UPDATE_MICROSECONDS = 10 * 1000 * 1000;

enum ecu_type
{
  accelerator,
  brake_pos,
  steering_wheel_angle,
  speed,
  rpm,
  dtc,
  gear_state,
  engine_load,
  mileage,
  driving_mode,
  operation_speed,
  emergency_stop
};

// locks
boost::mutex g_mutex_queue;
boost::mutex g_mutex_ros;
boost::mutex g_mutex_traffic_light;
boost::mutex g_mutex_event_1;
boost::mutex g_mutex_event_2;
boost::mutex g_mutex_mqtt;
boost::mutex g_mutex_sensor;
boost::mutex g_mutex_do;
boost::mutex g_mutex_fail_safe;
//boost::mutex mutex_serverStatus;

// ros queue
std::queue<std::string> g_ros_queue;
// adv heading queue
std::queue<std::string> g_obu_queue;

std::queue<std::string> g_vk_queue;
std::queue<std::string> g_vk_status_queue;

std::queue<json> g_mqtt_gnss_queue;
std::queue<json> g_mqtt_bsm_queue;
std::queue<json> g_mqtt_ecu_queue;
std::queue<json> g_mqtt_imu_queue;
std::queue<json> g_mqtt_sensor_queue;
std::queue<json> g_mqtt_detect_object_queue;
std::queue<std::string> g_mqtt_fail_safe_queue;

std::queue<std::string> g_traffic_light_queue;
std::queue<json> g_event_queue_1;
std::queue<json> g_event_queue_2;

TcpServer g_tcp_server;

msgs::DetectedObjectArray g_det_obj_array;
msgs::DetectedObjectArray g_tracking_obj_array;
msgs::LidLLA g_gps;
msgs::VehInfo g_veh_info;
json g_fps_json = { { "key", 0 } };
std::string g_vk102_response;
std::string g_mile_json;
std::string g_event_json;
std::string g_status_json;
msgs::RouteInfo g_route_info;
std::string g_board_list="00000000";
int g_route_id = 2000;

MqttClient g_mqtt_client;
bool g_is_mqtt_connected = false;

long g_event_recv_count = 0;
long g_event_send_count = 0;

const static double PI = 3.14;
// can data
double g_can_data[10] = { 0 };

// traffic light buffer
char g_taffic_light_buffer[1024];

std::string g_plate = "ITRI-ADV";
std::string g_vid = "vid";
const static int FPS_KEY_LEN = 27 + 16;
const static std::string keys[] = {
  "FPS_LidarAll",         "FPS_LidarDetection",   "FPS_camF_right",        "FPS_camF_center",     "FPS_camF_left",
  "FPS_camF_top",         "FPS_camR_front",       "FPS_camR_rear",         "FPS_camL_front",      "FPS_camL_rear",
  "FPS_camB_top",         "FPS_CamObjFrontRight", "FPS_CamObjFrontCenter", "FPS_CamObjFrontLeft", "FPS_CamObjFrontTop",
  "FPS_CamObjRightFront", "FPS_CamObjRightBack",  "FPS_CamObjLeftFront",   "FPS_CamObjLeftBack",  "FPS_CamObjBackTop",
  "FPS_current_pose",     "FPS_veh_info",         "FPS_dynamic_path_para", "FPS_Flag_Info01",     "FPS_Flag_Info02",
  "FPS_Flag_Info03",      "FPS_V2X_msg",          "FPS_camfront_bottom_60","FPS_camtop_close_120","FPS_camfront_top_far_30",
  "FPS_camleft_back_60",  "FPS_camleft_front_60", "FPS_camright_back_60",  "FPS_camright_front_60","FPS_camback_top_120",
  "FPS_cam_objfront_bottom_60","FPS_cam_objront_top_close_120","FPS_cam_objfront_top_far_30", "FPS_cam_objleft_back_60",
  "FPS_cam_objleft_front_60",  "FPS_cam_objright_back_60",     "FPS_cam_objright_front_60",   "FPS_cam_objback_top_120"
};


struct Pose
{
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

struct ArriveStop
{
  int id; // next stop id
  int status; // stauts
  int round; // current round
};

struct IMU
{
  double Gx;
  double Gy;
  double Gz;
  double Gyrox;
  double Gyroy;
  double Gyroz;
  double Ax;
  double Ay;
  double Az;
};

struct VehicelStatus
{
  
  float motor_temperature; //馬達
  float tire_pressure; //胎壓
  float air_pressure; //氣壓
  float battery; //電量 %
  float steer; //轉向
  float localization; //定位
  float odometry; //里程
  float speed; //車速 km/hr
  float rotating_speed; //轉速
  float bus_stop; //站點
  float vehicle_number; //車號
  float gear; //檔位
  int hand_brake; //手煞車
  float steering_wheel; //方向盤 0關1開
  int door; //車門 0關1開
  int air_conditioner; //空調
  float radar; //radar
  float lidar;
  float camera;
  float GPS;
  int headlight; //大燈 0關1開
  int wiper; //雨刷 0關 1開
  int indoor_light; //車內燈 0關 1開
  int gross_power; //總電源 0關 1開
  int left_turn_light; //左方向燈 0關 1開
  int right_turn_light;//右方向燈 0關 1開
  int estop; //E-Stop 0關 1開
  int ACC_state; //ACC電源狀態 0關 1開
  float time; //標準時間
  float driving_time; //行駛時間
  float mileage; //行駛距離
  float accelerator;
  float brake_pos;

};

struct BatteryInfo
{
  float gross_voltage;//總電壓 V
  float gross_current;//總電流 A
  float highest_voltage; //最高電池電壓 0.01V
  float highest_number; //最高電壓電池位置 電池編號
  float lowest_volage; //最低電池電壓 0.01V
  float lowest_number; //最低電壓電池位置 電池編號
  float voltage_deviation; //高低電壓叉 0.01V
  float highest_temp_location; //電池最高環境溫度位置 區域編號
  float highest_temperature; //電池最高環境溫度
};

unsigned int g_mode; //模式 自動/半自動/手動/鎖定
float g_emergency_exit; //緊急出口
  
Pose g_current_gnss_pose;
ArriveStop g_cuttent_arrive_stop;
IMU g_imu;

double g_base_mileage;
double g_delta_mileage;
std::string g_current_spat;

VehicelStatus g_vs;
BatteryInfo g_battery;

json genMqttGnssMsg();
json genMqttBmsMsg();
json genMqttECUMsg(ecu_type);
json genMqttIMUMsg();
json getMqttDOMsg();

/*=========================tools begin=========================*/
bool checkCommand(int argc, char** argv, const std::string& command)
{
  for (int i = 0; i < argc; i++)
  {
    if (command == argv[i])
    {
      return true;
    }
  }
  return false;
}

char* log_Time()
{
  struct tm* ptm;
  struct timeb stTimeb{};
  static char szTime[24];

  ftime(&stTimeb);
  ptm = localtime(&stTimeb.time);
  sprintf(szTime, "%04d-%02d-%02d %02d:%02d:%02d.%03d", ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday,
          ptm->tm_hour, ptm->tm_min, ptm->tm_sec, stTimeb.millitm);
  szTime[23] = 0;
  return szTime;
}

bool convertBoolean(int state)
{
  return state != 0;
}
/*=========================tools end=========================*/

/*========================= ROS callbacks begin=========================*/

void callback_flag_info04(const msgs::Flag_Info::ConstPtr& input)
{
    g_vs.steering_wheel =  input->Dspace_Flag04;
    g_vs.accelerator = input->Dspace_Flag05;
    g_vs.brake_pos = input->Dspace_Flag06;
    g_vs.rotating_speed = input->Dspace_Flag07;
}


void callback_detObj(const msgs::DetectedObjectArray& input)
{
  g_mutex_ros.lock();
  g_det_obj_array = input;
  g_mutex_ros.unlock();
}

void callback_gps(const msgs::LidLLA& input)
{
  std::cout << "callback gps " << std::endl;
  g_mutex_ros.lock();
  g_gps = input;
  g_mutex_ros.unlock();
}

void callback_veh(const msgs::VehInfo& input)
{
  g_mutex_ros.lock();
  g_veh_info = input;
  g_mutex_ros.unlock();
}

void callback_gnss2local(const geometry_msgs::PoseStamped::ConstPtr& input)
{
  g_mutex_ros.lock();
  tf::Quaternion gnss_q(input->pose.orientation.x, input->pose.orientation.y, input->pose.orientation.z,
                        input->pose.orientation.w);
  tf::Matrix3x3 gnss_m(gnss_q);
  g_current_gnss_pose.x = input->pose.position.x;
  g_current_gnss_pose.y = input->pose.position.y;
  g_current_gnss_pose.z = input->pose.position.z;
  gnss_m.getRPY(g_current_gnss_pose.roll, g_current_gnss_pose.pitch, g_current_gnss_pose.yaw);
  g_mutex_ros.unlock();
}

void callback_fps(const std_msgs::String::ConstPtr& input)
{
  g_mutex_ros.lock();
  std::string json_string = input->data.c_str();
  try
  {
    g_fps_json = json::parse(json_string);
  }
  catch (std::exception& e)
  {
    std::cout << "callback_fps message: " << e.what() << std::endl;
    return;
  }
  g_mutex_ros.unlock();
}

void callbackBusStopInfo(const msgs::Flag_Info::ConstPtr& input)
{
  //std::cout << "<<<<<<<<<<<<<<<callbackBusStopInfo>>>>>>>>>>>>>>>" << std::endl;
  g_board_list = "";
  float stop[8];
  memset(stop, 0, sizeof(stop));
  g_mutex_ros.lock();
  stop[0] = input->Dspace_Flag01;
  stop[1] = input->Dspace_Flag02;
  stop[2] = input->Dspace_Flag03;
  stop[3] = input->Dspace_Flag04;
  stop[4] = input->Dspace_Flag05;
  stop[5] = input->Dspace_Flag06;
  stop[6] = input->Dspace_Flag07;
  stop[7] = input->Dspace_Flag08;
  std::vector<int> stopids;
  for (int i = 0; i < 8; i++)
  {
    if (stop[i] == 1)
    {
      stopids.push_back(i + g_route_id + 1);
      g_board_list += "1";
    }else{
      g_board_list += "0";
    }
  }
  json j2;
  for (size_t j = 0; j < stopids.size(); j++)
  {
    int id = stopids[j];
    json obj;
    obj["stopid"] = id;
    j2.push_back(obj);
  }
  json j3;
  j3["msgInfo"] = "Success";
  j3["msgCode"] = 200;
  json j1;
  j1["messageObj"] = j3;
  j1["type"] = "M8.2.VK102";
  j1["plate"] = g_plate;
  j1["status"] = 0;
  j1["route_id"] = g_route_id;
  if (stopids.empty())
  {
    j1["bus_stops"] = json::array();
  }
  else
  {
    j1["bus_stops"] = j2;
  }

  g_vk102_response = j1.dump();
  g_mutex_ros.unlock();
}

void callbackNextStop(const msgs::Flag_Info::ConstPtr& input)
{
  g_mutex_ros.lock();
  g_cuttent_arrive_stop.id = g_route_id + (int)input->Dspace_Flag01;
  g_cuttent_arrive_stop.status = (int)input->Dspace_Flag02;
  //cuttent_arrive_stop.round = (int) input->PX2_Flag01;
  g_mutex_ros.unlock();
}

void callbackMileage(const std_msgs::String::ConstPtr& input)
{
  g_mutex_ros.lock();
  g_mile_json = input->data.c_str();
  std::cout << "mile info: " << g_mile_json << std::endl;

  g_mutex_ros.unlock();
}

void callbackRound(const std_msgs::Int32::ConstPtr& input)
{
  g_cuttent_arrive_stop.round = (int) input->data;
}

void callbackEvent(const std_msgs::String::ConstPtr& input)
{
  using namespace std;
  g_event_json = input->data.c_str();
  json j0 = json::parse(g_event_json);
  json j1;
  
  j1["type"] = "M8.2.VK003";
  j1["deviceid"] = g_plate;
  j1["lat"] = g_gps.lidar_Lat;
  j1["lng"] = g_gps.lidar_Lon;  
  j1["module"] = j0.at("module");
  j1["status"] = j0.at("status");
  j1["event_str"] = j0.at("event_str");
  j1["timestamp"] = j0.at("timestamp");
  
  
  if(g_event_queue_switch)
  {
    cout << " push to queue1 event: " << j1.dump() << endl;
    g_mutex_event_1.lock();
    g_event_queue_1.push(j1);
    g_mutex_event_1.unlock();
  }
  else
  {
    cout << " push to queue2 event: " << j1.dump() << endl;
    g_mutex_event_2.lock();
    g_event_queue_2.push(j1);
    g_mutex_event_2.unlock();
  }
  g_event_recv_count ++;
}

std::string get_msg_type(int id)
{
  switch (id)
  {
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

void callbackIMU(const sensor_msgs::Imu::ConstPtr& input)
{
  g_imu.Gx = input->linear_acceleration.x;
  g_imu.Gy = input->linear_acceleration.y;
  g_imu.Gz = input->linear_acceleration.z;
  g_imu.Gyrox = input->angular_velocity.x;
  g_imu.Gyroy = input->angular_velocity.y;
  g_imu.Gyroz = input->angular_velocity.z;
}

void callbackBI(const msgs::BackendInfo::ConstPtr& input)
{
  g_vs.motor_temperature = input->motor_temperature; //馬達溫度
  g_vs.tire_pressure = input->tire_pressure; //胎壓
  g_vs.air_pressure = input->air_pressure; //氣壓
  g_vs.battery =  input->battery; //電量 %
  g_vs.steer = input->steer; //轉向
  g_vs.localization = input->localization; //定位
  g_vs.odometry = input->odometry; //里程
  g_vs.speed = input->speed; //車速 km/hr
  //vs.rotating_speed = input->speed ; //轉速
  g_vs.bus_stop = input->bus_stop; //站點
  g_vs.vehicle_number = input->vehicle_number; //車號
  g_vs.gear = input->gear; //檔位
  g_vs.hand_brake = input->hand_brake; //手煞車
  //vs.steering_wheel = input->steering_wheel; //方向盤
  g_vs.door = input->door; //車門
  g_vs.air_conditioner = input->air_conditioner; //空調
  g_vs.radar = input->radar; //radar
  g_vs.lidar = input->lidar;
  g_vs.camera = input->camera;
  g_vs.GPS = input->GPS;
  g_vs.headlight = input->headlight; //大燈 0關1開
  g_vs.wiper = input->wiper; //雨刷 0關 1開
  g_vs.indoor_light = input->indoor_light; //車內燈 0關 1開
  g_vs.gross_power = input->gross_power; //總電源 0關 1開
  g_vs.left_turn_light = input->left_turn_light; //左方向燈 0關 1開
  g_vs.right_turn_light = input->right_turn_light;//右方向燈 0關 1開
  g_vs.estop = input->estop; //E-Stop 0關 1開
  g_vs.ACC_state = input->ACC_state; //ACC電源狀態 0關 1開
  g_vs.time = input->time; //標準時間
  g_vs.driving_time = input->driving_time; //行駛時間
  g_vs.mileage = input->mileage; //行駛距離
  g_battery.gross_voltage = input->gross_voltage; //總電壓 V
  g_battery.gross_current = input->gross_current; //總電流 A
  g_battery.highest_voltage = input->highest_voltage; //最高電池電壓 0.01V
  g_battery.highest_number = input->highest_number; //最高電壓電池位置 電池編號
  g_battery.lowest_volage = input->lowest_volage; //最低電池電壓 0.01V
  g_battery.lowest_number = input->lowest_number; //最低電壓電池位置 電池編號
  g_battery.voltage_deviation = input->voltage_deviation; //高低電壓差 0.01V
  g_battery.highest_temp_location = input->highest_temp_location; //電池最高環境溫度位置 區域編號
  g_battery.highest_temperature = input->highest_temperature; //電池最高環境溫度
  g_mode = input->mode; //模式 自動/半自動/手動/鎖定
  g_emergency_exit = input->emergency_exit; //緊急出口
  g_delta_mileage = input->mileage;
}


void callbackSersorStatus(const std_msgs::String::ConstPtr& input)
{
  using namespace std;
  g_mutex_sensor.lock();
  json j1 = json::parse(input->data.c_str());
  j1["vid"] = g_vid;
  g_mqtt_sensor_queue.push(j1.dump());
  g_mutex_sensor.unlock();
}

void callbackTracking(const msgs::DetectedObjectArray& input)
{
  g_mutex_do.lock();
  g_tracking_obj_array = input;
  g_mutex_do.unlock();
}

void callbackFailSafe(const std_msgs::String::ConstPtr& input)
{
  json J1 = json::parse(input->data.c_str());
  g_mutex_fail_safe.lock();
  g_mqtt_fail_safe_queue.push(J1.dump()); 
  g_mutex_fail_safe.unlock();
}

/*========================= ROS callbacks end =========================*/

/*========================= json parsers begin =========================*/
std::string get_jsonmsg_can(const std::string& type, double* data)
{
  // std::string time_string = log_Time ();
  json j1;
  j1["type"] = type;
  j1["plate"] = g_plate;
  j1["deviceID"] = "00:00:00:00:00:01";
  j1["dt"] = log_Time();
  if (type == "M8.2.adv002")
  {
    j1["speed"] = data[0];
    j1["front_brake_pressure"] = data[1];
    j1["rear_brake_pressure"] = data[2];
    j1["steering_wheel_angle"] = data[3];
  }
  return j1.dump();
}

std::string get_jsonmsg_ros(const std::string& type)
{
  json j1;
  j1["type"] = type;
  j1["plate"] = g_plate;
  j1["deviceID"] = "00:00:00:00:00:01";
  j1["dt"] = log_Time();
  if (type == "M8.2.adv001")
  {
    j1["lat"] = g_gps.lidar_Lat;
    j1["lon"] = g_gps.lidar_Lon;
    j1["speed"] = -1;
    j1["bearing"] = -1;
    j1["turn_signal"] = -1;
  }
  else if (type == "M8.2.adv002")
  {
    j1["speed"] = g_veh_info.ego_speed * 3.6;
    j1["front_brake_pressure"] = 0;
    j1["rear_brake_pressure"] = 0;
    j1["steering_wheel_angle"] = 0;
  }
  else if (type == "M8.2.adv003")
  {
    j1["sw_camera_signal"] = -1;
    j1["sw_lidar_signal"] = -1;
    j1["sw_radar_signal"] = -1;
    j1["slam"] = -1;
    j1["object_list"];
    int num = 0;
    for (size_t i = 0; i < g_det_obj_array.objects.size(); i++)
    {
      json j2;
      j2["object_adv_P0_x"] = g_det_obj_array.objects[i].bPoint.p0.x;
      j2["object_adv_P0_y"] = g_det_obj_array.objects[i].bPoint.p0.y;
      j2["object_adv_P1_x"] = g_det_obj_array.objects[i].bPoint.p1.x;
      j2["object_adv_P1_y"] = g_det_obj_array.objects[i].bPoint.p1.y;
      j2["object_adv_P2_x"] = g_det_obj_array.objects[i].bPoint.p2.x;
      j2["object_adv_P2_y"] = g_det_obj_array.objects[i].bPoint.p2.y;
      j2["object_adv_P3_x"] = g_det_obj_array.objects[i].bPoint.p3.x;
      j2["object_adv_P3_y"] = g_det_obj_array.objects[i].bPoint.p3.y;
      j2["object_adv_P4_x"] = g_det_obj_array.objects[i].bPoint.p4.x;
      j2["object_adv_P4_y"] = g_det_obj_array.objects[i].bPoint.p4.y;
      j2["object_adv_P5_x"] = g_det_obj_array.objects[i].bPoint.p5.x;
      j2["object_adv_P5_y"] = g_det_obj_array.objects[i].bPoint.p5.y;
      j2["object_adv_P6_x"] = g_det_obj_array.objects[i].bPoint.p6.x;
      j2["object_adv_P6_y"] = g_det_obj_array.objects[i].bPoint.p6.y;
      j2["object_adv_P7_x"] = g_det_obj_array.objects[i].bPoint.p7.x;
      j2["object_adv_P7_y"] = g_det_obj_array.objects[i].bPoint.p7.y;
      j2["object_adv_lat"] = -1;
      j2["object_adv_lon"] = -1;
      j2["object_adv_x"] = (g_det_obj_array.objects[i].bPoint.p0.x + g_det_obj_array.objects[i].bPoint.p7.x) / 2;
      j2["object_adv_y"] = (g_det_obj_array.objects[i].bPoint.p0.y + g_det_obj_array.objects[i].bPoint.p7.y) / 2;
      j2["object_type"] = g_det_obj_array.objects[i].classId;
      j2["object_status"] = -1;
      j2["object_length"] = -1;
      j1["object_list"] += j2;
      num++;
    }
    j1["object_count"] = num;
  }
  return j1.dump();
}

std::string get_jsonmsg_to_obu(const std::string& type)
{
  json j1;
  if (type == "M8.2.adv009")
  {
    j1["type"] = type;
    j1["lat"] = std::to_string(g_gps.lidar_Lat);
    j1["lon"] = std::to_string(g_gps.lidar_Lon);
    j1["speed"] = std::to_string(g_can_data[0]);
    j1["bearing"] = std::to_string(g_current_gnss_pose.yaw * 180 / PI);
  }
  return j1.dump();
}

std::string get_jsonmsg_to_vk_server(const std::string& type)
{
  json j1;
  j1["type"] = type;
  j1["deviceid"] = g_plate; //PLATE;
  j1["receivetime"] = log_Time();
  if (type == "M8.2.VK001")
  {
    j1["motor"] = g_vs.motor_temperature; // 馬達溫度 //2.1;
    j1["tirepressure"] =  g_vs.tire_pressure; //胎壓 //0.0;
    j1["airpressure"] = g_vs.air_pressure; //氣壓 //0.0;
    j1["electricity"] = g_vs.battery; //電量//0.0;
    j1["steering"] = g_vs.steer; // 轉向 
    j1["bearing"] = g_current_gnss_pose.yaw * 180 / PI;
    j1["heading"] = 0.0;
    j1["milage"] =  g_vs.odometry; //行駛距離//0.0;
    j1["speed"] = g_vs.speed; //vs.speed 車速 目前來源CAN
    j1["rotate"] = g_vs.rotating_speed; //轉速 //0.0;
    j1["gear"] = g_vs.gear; //檔位 //1;
    j1["handcuffs"] = convertBoolean(g_vs.hand_brake); //手煞車 //true;
    j1["Steeringwheel"] = g_vs.steering_wheel; //方向盤 //0.0;
    j1["door"] = convertBoolean(g_vs.door); //車門 //true;
    j1["airconditioner"] = convertBoolean(g_vs.air_conditioner); //空調;
    j1["lat"] = g_gps.lidar_Lat; //vs.location 目前來源 lidar_lla
    j1["lng"] = g_gps.lidar_Lon; //vs.location 目前來源 lidar_lla
    j1["headlight"] = convertBoolean(g_vs.headlight); //車燈 //true;
    j1["wiper"] =  convertBoolean(g_vs.wiper); //雨刷//true;
    j1["Interiorlight"] = convertBoolean(g_vs.indoor_light); //車內燈//true;
    j1["mainswitch"] = convertBoolean(g_vs.gross_power); //總電源//true;
    j1["leftlight"] = convertBoolean(g_vs.left_turn_light); //左方向燈; //true
    j1["rightlight"] = convertBoolean(g_vs.right_turn_light); //右方向燈//true;
    j1["EStop"] = convertBoolean(g_vs.estop); // E-Stop//true;
    j1["ACCpower"] = convertBoolean(g_vs.ACC_state); //ACC 電源//true;
    j1["ArrivedStop"] = g_cuttent_arrive_stop.id; //目前來源 NextStop/Info
    j1["ArrivedStopStatus"] = g_cuttent_arrive_stop.status; // 目前來源NextStop/Info
    j1["round"] = g_cuttent_arrive_stop.round; //目前來源 BusStop/Round
    j1["route_id"] = g_route_id;  //預設2000
    j1["RouteMode"] = 2;
    j1["Gx"] = g_imu.Gx;   //   目前來源 imu_data_rad
    j1["Gy"] = g_imu.Gy;   //   目前來源 imu_data_rad
    j1["Gz"] = g_imu.Gz;   //   目前來源 imu_data_rad
    j1["Gyrox"] = g_imu.Gyrox; // 目前來源 imu_data_rad
    j1["Gyroy"] = g_imu.Gyroy; // 目前來源 imu_data_rad
    j1["Gyroz"] = g_imu.Gyroz; // 目前來源 imu_data_rad
    j1["accelerator"] = g_can_data[4]; //無rostopic 目前來源CAN
    j1["brake_pedal"] = g_can_data[5]; //無rostopic 目前來源CAN
    j1["distance"] = 0.0; //? 跟mileage有何不同？
    j1["mainvoltage"] = g_battery.gross_voltage; //總電壓//0.0;
    j1["maxvoltage"] = g_battery.highest_voltage; //最高電池電壓//0.0;
    j1["maxvbatteryposition"] =  g_battery.highest_number; //最高電壓電池編號//"5517XW";
    j1["minvoltage"] = g_battery.lowest_volage; //最低電池電壓//0.0;
    j1["pressurediff"] = g_battery.voltage_deviation; //高低電壓差//0.0;
    j1["maxtbatteryposition"] = g_battery.lowest_number; //最低電池電壓 0.01V"454FG"; 
    j1["maxtemperature"] = g_battery.highest_temperature; //電池最高環境溫度//0.0;
    j1["Signal"] = g_current_spat; //無資料
    j1["CMS"] = 1; //無資料
    j1["setting"] = g_mode; // 自動/半自動/手動/鎖定
    j1["board_list"] = g_board_list;
  }
  else if (type == "M8.2.VK002")
  {
    j1["motor"] = g_vs.motor_temperature; // 馬達溫度 //2.1;
    j1["tirepressure"] = g_vs.tire_pressure; //胎壓 //0.0;
    j1["airpressure"] = g_vs.air_pressure; //氣壓 //0.0;
    j1["electricity"] =  g_vs.battery; //電量//0.0;
    j1["steering"] = g_vs.steer; // 轉向 
    j1["bearing"] = g_current_gnss_pose.yaw * 180 / PI;
    j1["heading"] = 0.0;
    j1["milage"] = g_vs.odometry; //行駛距離//0.0;
    j1["speed"] = g_vs.speed; //vs.speed 車速 
    j1["rotate"] = g_vs.rotating_speed; //轉速 //0.0;
    j1["gear"] = g_vs.gear; //檔位 //1;
    j1["handcuffs"] = convertBoolean(g_vs.hand_brake); //手煞車 //true;
    j1["Steeringwheel"] = g_vs.steering_wheel; //方向盤 //0.0;
    j1["door"] = convertBoolean(g_vs.door); //車門 //true;
    j1["airconditioner"] = convertBoolean(g_vs.air_conditioner); //空調;
    j1["lat"] = g_gps.lidar_Lat;  //vs.location 目前來源 lidar_lla
    j1["lng"] = g_gps.lidar_Lon;  //vs.location 目前來源 lidar_lla
    j1["headlight"] = convertBoolean(g_vs.headlight); //車燈 //true;
    j1["wiper"] = convertBoolean(g_vs.wiper); //雨刷//true;
    j1["Interiorlight"] = convertBoolean(g_vs.indoor_light); //車內燈//true;
    j1["mainswitch"] = convertBoolean(g_vs.gross_power); //總電源//true;
    j1["leftlight"] = convertBoolean(g_vs.left_turn_light); //左方向燈; //true
    j1["rightlight"] = convertBoolean(g_vs.right_turn_light); //右方向燈//true;
    j1["EStop"] = convertBoolean(g_vs.estop); // E-Stop//true;
    j1["ACCpower"] = convertBoolean(g_vs.ACC_state); //ACC 電源//true;
    j1["route_id"] = g_route_id; //default 2000
    j1["RouteMode"] = g_mode;
    j1["Gx"] = g_imu.Gx; //   目前來源 imu_data_rad
    j1["Gy"] = g_imu.Gy; //   目前來源 imu_data_rad
    j1["Gz"] = g_imu.Gz; //   目前來源 imu_data_rad
    j1["Gyrox"] = g_imu.Gyrox; //   目前來源 imu_data_rad
    j1["Gyroy"] = g_imu.Gyroy; //   目前來源 imu_data_rad
    j1["Gyroz"] = g_imu.Gyroz; //   目前來源 imu_data_rad
    j1["accelerator"] = g_can_data[4]; //無rostopic 目前來源CAN
    j1["brake_pedal"] = g_can_data[5]; //無rostopic 目前來源CAN
    j1["ArrivedStop"] = g_cuttent_arrive_stop.id; //目前來源 NextStop/Info
    j1["ArrivedStopStatus"] = g_cuttent_arrive_stop.status; //目前來源 NextStop/Info
    j1["round"] = g_cuttent_arrive_stop.round; //目前來源 BusStop/Round
    j1["Signal"] = g_current_spat; //無資料
    j1["CMS"] = 1; //無資料
    j1["setting"] = g_mode; 
    j1["EExit"] = g_emergency_exit; 
    j1["board_list"] = g_board_list;
  }else if (type == "M8.2.VK003"){
    j1["lat"] = g_gps.lidar_Lat;
    j1["lng"] = g_gps.lidar_Lon;
    json J0 = json::parse(g_event_json);
    j1["module"] = J0.at("module");
    j1["status"] = J0.at("status");
    j1["event_str"] = J0.at("event_str");
    j1["timestamp"] = J0.at("timestamp");
  }else if (type == "M8.2.VK004")
  {
    for (int i = 0; i < FPS_KEY_LEN; i++)
    {
      std::string key = keys[i];
      float value = g_fps_json.value(key, -1);
      j1[key] = value;
    }
  }
  else if (type == "M8.2.VK006")
  {
    // Roger 20200212 [ fix bug: resend the same json
    try
    {
      json j0 = json::parse(g_mile_json);
      j1["mileage_info"] = j0;

    }
    catch (std::exception& e)
    {
      //std::cout << "mileage: " << e.what() << std::endl;
    }
    
    g_mile_json = "";
    // Roger 20200212 ]
  }
  return j1.dump();
}
/*========================= json parsers end =========================*/

/*========================= thread runnables begin =========================*/
void mqtt_pubish(std::string msg)
{
  if(g_is_mqtt_connected){
      std::string topic = "vehicle/report/" + g_vid;
      std::cout << "publish "  << msg << std::endl;
      g_mqtt_client.publish(topic, msg);
    }
}

void sendRun(int argc, char** argv)
{
  using namespace std;
  UdpClient udp_back_client;
  UdpClient udp_obu_client;
  UdpClient udp_vk_client;
  UdpClient udp_tablet_client;
  UdpClient udp_vk_fg_client;
  UdpClient udp_vk_fail_safe_client;

  udp_back_client.initial(UDP_AWS_SRV_ADRR, UDP_AWS_SRV_PORT);
  udp_obu_client.initial(UDP_OBU_ADRR, UDP_OBU_PORT);
  udp_vk_client.initial(UDP_VK_SRV_ADRR, UDP_VK_SRV_PORT);
  udp_tablet_client.initial("192.168.1.3", 9876);
  udp_vk_fg_client.initial("140.134.128.42", 8888);
  udp_vk_fail_safe_client.initial(UDP_VK_SRV_ADRR, 55554);

  // UDP_VK_client.initial("192.168.43.24", UDP_VK_SRV_PORT);
  while (true)
  {
    g_mutex_queue.lock();
    while (!g_ros_queue.empty())
    {
      udp_back_client.send_obj_to_server(g_ros_queue.front(), g_flag_show_udp_send);
      g_ros_queue.pop();
    }

    while (!g_obu_queue.empty())
    {
      udp_obu_client.send_obj_to_server(g_obu_queue.front(), g_flag_show_udp_send);
      g_obu_queue.pop();
    }

    while (!g_vk_queue.empty())
    {
      udp_vk_client.send_obj_to_server(g_vk_queue.front(), g_flag_show_udp_send);
      //UDP_TABLET_client.send_obj_to_server(vkQueue.front(), flag_show_udp_send);
      g_vk_queue.pop();
    }
    
    while (!g_vk_status_queue.empty())
    {
      udp_vk_client.send_obj_to_server(g_vk_status_queue.front(), true);
      udp_vk_fg_client.send_obj_to_server(g_vk_status_queue.front(), true);
      udp_tablet_client.send_obj_to_server(g_vk_status_queue.front(), g_flag_show_udp_send);
      g_vk_status_queue.pop();
    }
    g_mutex_queue.unlock();

    g_mutex_mqtt.lock();
    json j1;
    std::string states;
    json detectObject;
    //std::string vid = "dc5360f91e74";
    j1["vid"] = g_vid;
    json gnss_list = json::array();
    json bsm_list = json::array();
    json ecu_list = json::array();
    json imu_list = json::array();
    if(!g_mqtt_gnss_queue.empty())
    {
      while(!g_mqtt_gnss_queue.empty())
      {
        json gnss = g_mqtt_gnss_queue.front();
        gnss_list.push_back(gnss);
        g_mqtt_gnss_queue.pop();
      }
      j1["gnss"] = gnss_list;
    }

    if(!g_mqtt_bsm_queue.empty())
    {
      while(!g_mqtt_bsm_queue.empty()){
        json bsm = g_mqtt_bsm_queue.front();
        bsm_list.push_back(bsm);
        g_mqtt_bsm_queue.pop();
      }
      j1["bms"] = bsm_list;
    }

    if(!g_mqtt_ecu_queue.empty())
    {
      while(!g_mqtt_ecu_queue.empty()){
        json ecu = g_mqtt_ecu_queue.front();
        ecu_list.push_back(ecu);
        g_mqtt_ecu_queue.pop();
      }
      j1["ecu"] = ecu_list;
    }

    if(!g_mqtt_imu_queue.empty())
    {
      while(!g_mqtt_imu_queue.empty()){
        json jimu = g_mqtt_imu_queue.front();
        imu_list.push_back(jimu);
        g_mqtt_imu_queue.pop();
      }
      j1["imu"] = imu_list;
    }

    while(!g_mqtt_sensor_queue.empty()){
       g_mutex_sensor.lock();
       states = g_mqtt_sensor_queue.front();
       g_mqtt_sensor_queue.pop();
       g_mutex_sensor.unlock();
       mqtt_pubish(states);
    }

    while(!g_mqtt_detect_object_queue.empty()){
        g_mutex_do.lock();
        json json_detect_object;
        detectObject = g_mqtt_detect_object_queue.front();
        json_detect_object["vid"] = g_vid;
        json_detect_object["DO"] = detectObject;
        g_mqtt_detect_object_queue.pop();
        g_mutex_do.unlock();
        mqtt_pubish(json_detect_object.dump());
    }

    while(!g_mqtt_fail_safe_queue.empty())
    {
	    g_mutex_fail_safe.lock();
        std::string fail_safe = g_mqtt_fail_safe_queue.front();
        json j1 = json::parse(fail_safe);
        j1["type"] = "M8.2.VK003.2";
        j1["deviceid"] = g_plate;
	    g_mqtt_fail_safe_queue.pop();
	    g_mutex_fail_safe.unlock();
        udp_vk_fail_safe_client.send_obj_to_server(j1.dump(), true);
    }

    mqtt_pubish(j1.dump());
    g_mutex_mqtt.unlock();


    if(g_event_queue_switch)
    {
      if(!g_event_queue_1.empty()){
        g_mutex_event_1.lock();
        g_event_queue_switch = false;
      
        while (!g_event_queue_1.empty())
        {
          json j = g_event_queue_1.front();
          string jstr = j.dump();
          cout << "++++++++++++++++++++++++++++++send from q 1 " << jstr << endl;
          udp_vk_client.send_obj_to_server(jstr, g_flag_show_udp_send);
          udp_tablet_client.send_obj_to_server(jstr, g_flag_show_udp_send);
          g_event_queue_1.pop();
        }

        g_mutex_event_1.unlock();
      }  
    }//if(event_queue_switch)
    else
    {
      if(!g_event_queue_2.empty()){
        g_mutex_event_2.lock();
        g_event_queue_switch = true;
    
        while (!g_event_queue_2.empty())
        {
         
          json j = g_event_queue_2.front();
          string jstr = j.dump();
          cout << "+++++++++++++++++++++++++++++++send from q 2 " << jstr << endl;
          udp_vk_client.send_obj_to_server(jstr, g_flag_show_udp_send);
          udp_tablet_client.send_obj_to_server(jstr, g_flag_show_udp_send);
          g_event_queue_2.pop();
        }

        g_mutex_event_2.unlock();
      }//if(eventQueue2.size() != 0)
    }//else
    //cout << " receive event: " << event_recv_count << endl;
    //cout << " send event: " << event_send_count << endl;
    boost::this_thread::sleep(boost::posix_time::microseconds(100*1000));
  }
}

void receiveCanRun(int argc, char** argv)
{
  CanReceiver receiver;
  while (true)
  {
    receiver.initial();
    for (int i = 0; i < 1; i++)
    {
      int msg_id = receiver.receive(g_can_data);
      std::string type = get_msg_type(msg_id);
      std::string temp = get_jsonmsg_can(type, g_can_data);
      g_mutex_queue.lock();
      g_ros_queue.push(temp);
      g_mutex_queue.unlock();
    }
    receiver.closeSocket();
    boost::this_thread::sleep(boost::posix_time::microseconds(500000));
  }
}

void receiveUDPRun(int argc, char** argv)
{
  while (true)
  {
    UdpServer udp_obu_server(UDP_ADV_SRV_ADRR, UDP_ADV_SRV_PORT);
    //UdpServer UDP_OBU_server("192.168.43.204", UDP_ADV_SRV_PORT);
    int result = udp_obu_server.recv(g_taffic_light_buffer, sizeof(g_taffic_light_buffer));
    g_current_spat = "";
    if (result != -1)
    {
      g_mutex_traffic_light.lock();
      std::string temp_str(g_taffic_light_buffer);
      g_current_spat = temp_str;
      memset(g_taffic_light_buffer, 0, sizeof(g_taffic_light_buffer));
      g_traffic_light_queue.push(temp_str);
      g_mutex_traffic_light.unlock();
    }

    // 100 ms
    boost::this_thread::sleep(boost::posix_time::microseconds(UDP_SERVER_UPDATE_MICROSECONDS));
  }
}

void sendROSRun(int argc, char** argv)
{
  while (ros::ok())
  {
    g_mutex_traffic_light.lock();
    while (!g_traffic_light_queue.empty())
    {
      std::string traffic_msg = g_traffic_light_queue.front();
      g_traffic_light_queue.pop();
      msgs::Spat spat;
      json j0 = json::parse(traffic_msg);
      try
      {
        json j1 = j0.at("SPaT_MAP_Info");
        spat.lat = j1.at("Latitude");
        spat.lon = j1.at("Longitude");
        spat.spat_state = j1.at("Spat_state");
        spat.spat_sec = j1.at("Spat_sec");
        spat.signal_state = j1.at("Signal_state");
        spat.index = j1.at("Index");
      } 
      catch(std::exception& e)
      {
        std::cout << "parsing fail: " << e.what() << " "<<std::endl;
      }
      //send traffic light
      RosModuleTraffic::publishTraffic(TOPIC_TRAFFIC, spat);
    }
    g_mutex_traffic_light.unlock();
    
    //send route info
    RosModuleTraffic::publishRoute(TOPIC_ROUTE, g_route_info);

    boost::this_thread::sleep(boost::posix_time::microseconds(500000));
    ros::spinOnce();
  }
}

void receiveRosRun(int argc, char** argv)
{
  bool is_big_bus = checkCommand(argc, argv, "-big");
  bool is_new_map = checkCommand(argc, argv, "-newMap");

  RosModuleTraffic::RegisterCallBack(callback_detObj, callback_gps, callback_veh, callback_gnss2local, callback_fps,
                                     callbackBusStopInfo, callbackMileage, callbackNextStop, callbackRound, callbackIMU, 
                                     callbackEvent, callbackBI, callbackSersorStatus,callbackTracking,callbackFailSafe,               
                                     callback_flag_info04, is_new_map);


  while (ros::ok())
  {
    g_mutex_ros.lock();
    std::string temp_adv001 = get_jsonmsg_ros("M8.2.adv001");
    g_mutex_queue.lock();
    g_ros_queue.push(temp_adv001);

    g_mutex_queue.unlock();

    std::string temp_adv003 = get_jsonmsg_ros("M8.2.adv003");
    g_mutex_queue.lock();
    g_ros_queue.push(temp_adv003);
    g_mutex_queue.unlock();

    std::string temp_adv009 = get_jsonmsg_to_obu("M8.2.adv009");
    g_mutex_queue.lock();
    g_obu_queue.push(temp_adv009);
    g_mutex_queue.unlock();

    if (is_big_bus)
    {
      std::string temp_vk002 = get_jsonmsg_to_vk_server("M8.2.VK002");
      g_mutex_queue.lock();
      g_vk_status_queue.push(temp_vk002);
      g_mutex_queue.unlock();
    }
    else
    {
      std::string temp_vk001 = get_jsonmsg_to_vk_server("M8.2.VK001");
      g_mutex_queue.lock();
      g_vk_status_queue.push(temp_vk001);
      g_mutex_queue.unlock();
    }

    std::string temp_vk004 = get_jsonmsg_to_vk_server("M8.2.VK004");
    g_mutex_queue.lock();
    g_vk_queue.push(temp_vk004);
    g_mutex_queue.unlock();

    std::string temp_vk_006 = get_jsonmsg_to_vk_server("M8.2.VK006");
    g_mutex_queue.lock();
    g_vk_queue.push(temp_vk_006);
    g_mutex_queue.unlock();


    g_mutex_mqtt.lock();
    json gnssobj = genMqttGnssMsg();
    json bsmobj  = genMqttBmsMsg();
    json ecu_acc_obj = genMqttECUMsg(ecu_type::accelerator);
    json ecu_brk_obj = genMqttECUMsg(ecu_type::brake_pos);
    json ecu_speed_obj = genMqttECUMsg(ecu_type::speed);
    json ecu_steer_obj = genMqttECUMsg(ecu_type::steering_wheel_angle);
    json ecu_geer_obj = genMqttECUMsg(ecu_type::gear_state);
    json ecu_rpm_obj = genMqttECUMsg(ecu_type::rpm);
    json ecu_engineload_obj = genMqttECUMsg(ecu_type::engine_load);
    json ecu_dtc_obj = genMqttECUMsg(ecu_type::dtc);
    json ecu_mileage_obj = genMqttECUMsg(ecu_type::mileage);
    json ecu_mode_obj = genMqttECUMsg(ecu_type::driving_mode);
    json ecu_operation_speed_obj = genMqttECUMsg(ecu_type::operation_speed);
    json ecu_emergency_stop_obj = genMqttECUMsg(ecu_type::emergency_stop);
    json imu_obj = genMqttIMUMsg();
    json detect_obj = getMqttDOMsg();


    g_mqtt_gnss_queue.push(gnssobj);
    g_mqtt_bsm_queue.push(bsmobj);

    g_mqtt_ecu_queue.push(ecu_acc_obj);
    g_mqtt_ecu_queue.push(ecu_brk_obj);
    g_mqtt_ecu_queue.push(ecu_speed_obj);
    g_mqtt_ecu_queue.push(ecu_steer_obj);
    g_mqtt_ecu_queue.push(ecu_geer_obj);
    g_mqtt_ecu_queue.push(ecu_rpm_obj);
    g_mqtt_ecu_queue.push(ecu_mileage_obj);
    g_mqtt_ecu_queue.push(ecu_mode_obj);
    g_mqtt_ecu_queue.push(ecu_operation_speed_obj);
    g_mqtt_ecu_queue.push(ecu_engineload_obj);
    g_mqtt_ecu_queue.push(ecu_dtc_obj);

    g_mqtt_imu_queue.push(imu_obj);

    g_mqtt_detect_object_queue.push(detect_obj);

    g_mutex_mqtt.unlock();


    g_mutex_ros.unlock();

    boost::this_thread::sleep(boost::posix_time::microseconds(ROS_UPDATE_MICROSECONDS));
    ros::spinOnce();
  }
}

void getServerStatusRun(int argc, char** argv)
{
  try
  {
    size_t buff_size = 2048;
    char buffer_f[buff_size];
    memset(buffer_f, 0, sizeof(buffer_f));
    TCPClient tcp_vk_client;
    tcp_vk_client.initial(TCP_VK_SRV_ADRR, TCP_VK_SRV_PORT);
    // TCP_VK_client.initial("192.168.43.24", 8765);
    tcp_vk_client.connectServer();
    json j1;
    j1["type"] = "M8.2.VK008";
    j1["deviceid"] = "ITRI-ADV";
    std::string json_str = j1.dump();
    const char* msg = json_str.c_str();
    tcp_vk_client.sendRequest(msg, strlen(msg));
    tcp_vk_client.recvResponse(buffer_f, buff_size);
    std::string response(buffer_f);
    std::cout << "=======Response: " << response << std::endl;
    json j2;
    j2 = json::parse(response);
    g_base_mileage = j2["totle_delta_km"];

    // connect to server success.
    RosModuleTraffic::publishServerStatus(TOPIC_SERCER_STATUS, true);
  }
  catch (std::exception& e)
  {
    std::cout << "getServerStatus message: " << e.what() << std::endl;
    // connect to server fail.
    RosModuleTraffic::publishServerStatus(TOPIC_SERCER_STATUS, false);
  }
}

std::string genResMsg(int status)
{
  json j1;
  json j2;

  j2["msgInfo"] = "Success";
  j2["msgCode"] = 200;
  j1["messageObj"] = j2;
  j1["status"] = status;
  return j1.dump();
}

std::string genErrorMsg(int code, std::string msg)
{
  json j1;
  json j2;

  j2["msgInfo"] = msg;
  j2["msgCode"] = code;
  j1["messageObj"] = j2;
  j1["type"] = "M8.2.VK102";
  j1["plate"] = g_plate;
  j1["status"] = 0;
  j1["route_id"] = g_route_id;
  j1["bus_stops"] = json::array();
  return j1.dump();
}

/*
bool checkStopID(unsigned short in_stop_id, unsigned short out_stop_id)
{
  if ((in_stop_id < 0) | (out_stop_id < 0))
    return false;
  return true;
}
*/

// response
void VK102callback(const std::string& request)
{
  using namespace std;
  json J1;
  unsigned int in_round;
  unsigned int out_round;
  unsigned int in_stopid;
  unsigned int out_stopid;
  std::string type;

  // clear response
  g_vk102_response = "";

  // parsing
  try
  {
    J1 = json::parse(request);
  }
  catch (std::exception& e)
  {
    std::cout << "VK102callback message: " << e.what() << std::endl;
    // 400 bad request
    g_tcp_server.send_json(genErrorMsg(400, e.what()));
    return;
  }

  std::cout << "VK102callback J1: " << J1.dump() << std::endl;

  // get data
  try
  {
    type = J1.at("type").get<std::string>();
    in_stopid = J1.at("in_stopid").get<unsigned int>();
    out_stopid = J1.at("out_stopid").get<unsigned int>();
    in_round = J1.at("in_round").get<unsigned int>();
    out_round = J1.at("out_round").get<unsigned int>();
  }
  catch (std::exception& e)
  {
    std::cout << "VK102callback message: " << e.what() << std::endl;
    g_tcp_server.send_json(genErrorMsg(400, e.what()));
    return;
  }

  // check stop id
  /*
  if (!checkStopID(in_stopid, out_stopid))
  {
    std::cout << "check id fail " << std::endl;
    server.send_json(genErrorMsg(422, "bad stop id."));
    return;
  }
  */
  //char msg[36];
  //sprintf(msg, "%d#%d", in_stopid, out_stopid);
  msgs::StopInfoArray reserve;
  msgs::StopInfo in_stop_info;
  msgs::StopInfo out_stop_info;
  
  in_stop_info.round = in_round;
  in_stop_info.id = in_stopid;
  out_stop_info.round = out_round;
  out_stop_info.id = out_stopid;

  reserve.stops.push_back(in_stop_info);
  reserve.stops.push_back(out_stop_info);
  
  RosModuleTraffic::publishReserve(TOPIC_RESERVE, reserve);
  // 300 millis seconds
  //boost::this_thread::sleep(boost::posix_time::microseconds(REVERSE_SLEEP_TIME_MICROSECONDS));
  //std::cout << "wake up, VK102Response: " << VK102Response << std::endl;

  /* check response from /BusStop/Info */ 
  unsigned short retry_count = 0;
  while ( g_vk102_response.empty() && (retry_count < RESERVE_WAITING_TIMEOUT / REVERSE_SLEEP_TIME_MICROSECONDS ) )
  {
    retry_count ++;
    boost::this_thread::sleep(boost::posix_time::microseconds(REVERSE_SLEEP_TIME_MICROSECONDS));
  }
  
  /* response to server */
  if (g_vk102_response.empty())
  {
    g_tcp_server.send_json(genErrorMsg(201, "No data from /BusStop/Info."));
  }else {
    g_tcp_server.send_json(g_vk102_response);
  }
}


// response
void VK103callback(json reqJson)
{
  using namespace std;
  
  vector<unsigned int> stopids;
  
  // clear response
  g_vk102_response = "";
 
  cout << "VK103callback reqJson: " << reqJson.dump() << endl;

  // get data
  try
  {
    stopids = reqJson.at("stopid").get< vector<unsigned int> >();
  }
  catch (exception& e)
  {
    cout << "VK103callback message: " << e.what() << endl;
    g_tcp_server.send_json(genErrorMsg(400, e.what()));
    return;
  }

  msgs::StopInfoArray reserve;
  for (size_t i = 0 ; i < stopids.size(); i++)
  {
    msgs::StopInfo stop;
    stop.round = 1;
    stop.id = stopids[i];
    reserve.stops.push_back(stop);
  }
  cout << "VK103callback msgs for ros: " <<  endl;

  RosModuleTraffic::publishReserve(TOPIC_RESERVE, reserve);
  // 300 millis seconds
  //boost::this_thread::sleep(boost::posix_time::microseconds(REVERSE_SLEEP_TIME_MICROSECONDS));
  //std::cout << "wake up, VK102Response: " << VK102Response << std::endl;

  /* check response from /BusStop/Info */ 
  unsigned short retryCount = 0;
  while ( g_vk102_response.empty() && (retryCount < RESERVE_WAITING_TIMEOUT / REVERSE_SLEEP_TIME_MICROSECONDS ) )
  {
    retryCount ++;
    std::cout << "retry: " << retryCount << std::endl;
    boost::this_thread::sleep(boost::posix_time::microseconds(REVERSE_SLEEP_TIME_MICROSECONDS));
  }
  
  /* response to server */
  if (g_vk102_response.empty())
  {
    g_tcp_server.send_json(genErrorMsg(201, "No data from /BusStop/Info."));
  }else {
    g_tcp_server.send_json(g_vk102_response);
  }
}

void VK104callback(json reqJson)
{
   using namespace std;
   
   cout << "VK104callback reqJson: " << reqJson.dump() << endl;
  
   string route_path;
   vector<unsigned int> stopids;
   

   // get data
   try
   {
     g_route_id = reqJson.at("routeid").get<int>();
     route_path = reqJson.at("routepath").get<string>(); 
     stopids = reqJson.at("stopid").get< vector<unsigned int> >();
   }
   catch (exception& e)
   {
     cout << "VK104callback message: " << e.what() << endl;
     g_tcp_server.send_json(genResMsg(0));
     return;
   }
 
   g_route_info.routeid = g_route_id;
   g_route_info.route_path = route_path;
   g_route_info.stops.clear();
   for (size_t i = 0 ; i < stopids.size(); i++)
   {
     msgs::StopInfo stop;
     stop.round = 1;
     stop.id = stopids[i];
     g_route_info.stops.push_back(stop);
   }

    /* check response from /BusStop/Info */ 
   unsigned short retry_count = 0;
   while ( g_vk102_response.empty() && (retry_count < RESERVE_WAITING_TIMEOUT / REVERSE_SLEEP_TIME_MICROSECONDS ) )
   {
     retry_count ++;
     boost::this_thread::sleep(boost::posix_time::microseconds(REVERSE_SLEEP_TIME_MICROSECONDS));
   }
  
   /* response to server */
   if (g_vk102_response.empty())
   {
     g_tcp_server.send_json(genResMsg(0));
   }else {
     g_tcp_server.send_json(genResMsg(1));
   }
}

//route api
void route(const std::string &request)
{
  using namespace std;
  string type;
  json requestJson;

  // parsing
  try
  {
    requestJson = json::parse(request);
  }
  catch (exception& e)
  {
    cout << "tcp server callback message: " << e.what() << endl;
    // 400 bad request
    g_tcp_server.send_json(genErrorMsg(400, e.what()));
    return;
  }

  // get type
  try
  {
    type = requestJson.at("type").get<string>();
  }
  catch (std::exception& e)
  {
    std::cout << "tcp server callback message: " << e.what() << std::endl;
    g_tcp_server.send_json(genErrorMsg(400, e.what()));
    return;
  }

  if ("M8.2.VK102" == type)
  {
    VK102callback(request);
  } else if ("M8.2.VK103" == type)
  {
    VK103callback(requestJson);
  } else if ("M8.2.VK104" == type)
  {
    VK104callback(requestJson);
  }
}


// start TCP server to receive VK102 reserve bus from backend.
void tcpServerRun(int argc, char** argv)
{
  // set ip and port
  g_tcp_server.initial(TCP_ADV_SRV_ADRR, TCP_ADV_SRV_PORT);
  //server.initial("192.168.43.204",8765);
  // server.initial("192.168.2.110",8765);
  // listening connection request
  int result = g_tcp_server.start_listening();
  
  if (result >= 0)
  {
    // accept and read request and handle request in VK102callback.
    try
    {
      g_tcp_server.wait_and_accept(route);
    }
    catch (std::exception& e)
    {
      g_tcp_server.send_json(genErrorMsg(408, "You should send request in 10 seconds after you connected to ADV."));
    }
  }
}

double get_GNSS_heading_360(double heading)
{
    double result = -1;
    if(heading < 0){
        result = heading + 360;
    }else {
        result = heading; 
    }
    return result;
}

json genMqttGnssMsg()
{
  using namespace std::chrono;
  json gnss;
  uint64_t timestamp_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  //uint64_t source_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
  double lat = g_gps.lidar_Lat;
  double lon = g_gps.lidar_Lon;
  double alt = g_gps.lidar_Alt;
  gnss["coord"] = {lat, lon, alt};
  //gnss["speed"] = -1; remove speed
  gnss["heading"] = get_GNSS_heading_360(g_current_gnss_pose.yaw * 180 / PI);
  gnss["timestamp"] = timestamp_ms;
  gnss["source_time"] = timestamp_ms;
  return gnss;
}

json genMqttBmsMsg()
{
  using namespace std::chrono;
  uint64_t timestamp_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  json bsm;
  bsm["uid"] = g_plate;
  bsm["current"] = g_battery.gross_current;
  bsm["voltage"] = g_battery.gross_voltage;
  bsm["capacity"] =g_vs.battery;
  bsm["design_capacity"] = -1 ;
  bsm["timestamp"] = timestamp_ms;
  bsm["source_time"] = timestamp_ms;
  return bsm;
}

json genMqttECUMsg(ecu_type type)
{

  using namespace std::chrono;
  uint64_t timestamp_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  json ecu;

  switch(type){
    case ecu_type::accelerator:
      ecu["accelerator_pos"] = g_vs.accelerator;
      break;
    case ecu_type::brake_pos:
      ecu["brake_pos"] = g_vs.brake_pos;
      break;
    case ecu_type::steering_wheel_angle:
      ecu["steering_wheel_angle"] = g_vs.steering_wheel;
      break;
    case ecu_type::speed:
      ecu["speed"] = g_vs.speed;
      break;
    case ecu_type::rpm:
      ecu["rpm"] = g_vs.rotating_speed;
      break;
    case ecu_type::gear_state:
      ecu["gear_state"] = "1";
      break;
    case ecu_type::dtc:
      ecu["dtc"] = "test";
      break;
    case ecu_type::engine_load:
      ecu["engine_load"] = -1.0;
      ecu["vin"] = "";
      break;
    case ecu_type::mileage:
      ecu["total_mileage"] = g_base_mileage + g_delta_mileage;
      break;
    case ecu_type::operation_speed:
      ecu["operation_speed"] = -1.0;
      ecu["maximum_speed"] = 35;
      break;
    case ecu_type::driving_mode:
      ecu["driving_mode"] = 1;
      break;
    case ecu_type::emergency_stop:
      ecu["emergency_stop"] = g_vs.estop;
      break;

  }
  ecu["timestamp"] = timestamp_ms;
  ecu["source_time"] = timestamp_ms;
  return ecu;
}

json genMqttIMUMsg()
{
  using namespace std::chrono;
  uint64_t timestamp_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  json jimu;
  jimu["uid"] = g_plate;
  //jimu["gyro_x"] = imu.Gyrox;
  //jimu["gyro_y"] = imu.Gyroy;
  //jimu["gyro_z"] = imu.Gyroz;
  jimu["gyro"] = {g_imu.Gyrox, g_imu.Gyroy, g_imu.Gyroz};
  jimu["roll_rate"] = g_current_gnss_pose.roll;
  jimu["pitch_rate"] = g_current_gnss_pose.pitch;
  jimu["yaw_rate"] = g_current_gnss_pose.yaw;
  //jimu["acc_x"] = imu.Gx;
  //jimu["acc_y"] = imu.Gy;
  //jimu["acc_z"] = imu.Gz;
  //jimu["acc"] = {imu.Gx, imu.Gy, imu.Gz};
  //jimu["d2xdt"] = -1.0;
  //jimu["d2ydt"] = -1.0;
  //jimu["d2zdt"] = -1.0;
  jimu["timestamp"] = timestamp_ms;
  jimu["source_time"] = timestamp_ms;
  return jimu;
}

json getMqttDOMsg(){
    using namespace std::chrono;
    uint64_t timestamp_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    json detect;
    json detect_obj;
    json obj_array;
    for (size_t i = 0; i < g_tracking_obj_array.objects.size(); i++)
    {
         json obj;
         obj["classification"] = g_tracking_obj_array.objects[i].classId;
         obj["tid"] = g_tracking_obj_array.objects[i].track.id;
         obj["do_coordinate"] = {    
            g_tracking_obj_array.objects[i].center_point_gps.x,
            g_tracking_obj_array.objects[i].center_point_gps.y, 
            g_tracking_obj_array.objects[i].center_point_gps.z 
         };
         obj["do_heading"] = {
            g_tracking_obj_array.objects[i].heading_enu.x,
            g_tracking_obj_array.objects[i].heading_enu.y,
            g_tracking_obj_array.objects[i].heading_enu.z,
            g_tracking_obj_array.objects[i].heading_enu.w,
         };
         obj["do_dimension"] = {
             g_tracking_obj_array.objects[i].dimension.length,
             g_tracking_obj_array.objects[i].dimension.width,
             g_tracking_obj_array.objects[i].dimension.height,
         };
         obj_array.push_back(obj);
    }
    if (obj_array.empty()){
        detect_obj["obj"] = json::array();
    }else{
        detect_obj["obj"] = obj_array;
    }
    detect_obj["timestamp"] = timestamp_ms;
    detect_obj["source_time"] = timestamp_ms;
    return detect_obj;

}

static void on_mqtt_connect(struct mosquitto* client, void* obj, int rc)
{
  std::string result;
  std::string topic = "vehicle/report/" + g_vid;
  switch (rc)
  {
    case 0:
      result = ": success";
      g_is_mqtt_connected = true;
      break;
    case 1:
      result = ": connection refused (unacceptable protocol version)";
      break;
    case 2:
      result = ": connection refused (identifier rejected)";
      break;
    case 3:
      result = ": connection refused (broker unavailable)";
      break;
    default:
      result = ": unknown error";
  }
  std::cout << "on_Connect result= " << rc << result << std::endl;
}

void mqttPubRun(int argc, char** argv)
{
  g_mqtt_client.setOnConneclCallback(on_mqtt_connect);
  g_mqtt_client.vid = g_vid;
  g_mqtt_client.connect();
}

/*========================= thread runnables end =========================*/

int main(int argc, char** argv)
{
  using namespace std;
  RosModuleTraffic::Initial(argc, argv);
  g_plate = RosModuleTraffic::getPlate();
  g_vid = RosModuleTraffic::getVid();
  //RosModuleTraffic::advertisePublisher();
  /*Start thread to receive data from can bus.*/
  if (!checkCommand(argc, argv, "-no_can"))
  {
    boost::thread thread_can_receive(receiveCanRun, argc, argv);
  }

  /*Start thread to receive data from ros topics:
    1. LidarDetection: Lidar objects.
    2. lidar_lla : ADV location for backend.
    3. veh_info: vehicle infomation for backend.
    4. gnss2local_data: ADV heading for OBU.
    5. /GUI/topic_fps_out : GUI fps infomation for backend.
    6. /BusStop/Info : For backend reserve system.
    7. /reserve/request: Debug .
  */
  boost::thread thread_ros_receive(receiveRosRun, argc, argv);

  /*start thread for UDP client :
    1. AWS backend: API: adv001 adv003
    2. VK(.120) backenda: API: VK001 VK002 VK004
    3. OBU. API:adv009
  */
  boost::thread thread_send(sendRun, argc, argv);

  /*Start thread for UDP server to receive traffic light infomation from OBU. */
  if (checkCommand(argc, argv, "-udp_srv"))
  {
    g_flag_show_udp_send = false;
    boost::thread thread_udp_receive(receiveUDPRun, argc, argv);
  }
  /*Start thread to publish ROS topics:
    1. /traffic : traffic light from OBU to GUI.
    2. /serverStatus : server status for ADV_op/sys_ready.
    3. /reserve/request: Reserve request from back end to Control team.
  */
  boost::thread thread_ros_send(sendROSRun, argc, argv);

  /*Strart thread for TCP client: Send VK005 to get server status.*/
  boost::thread thread_get_server_status(getServerStatusRun, argc, argv);

  /*Startr thread for TCP server: Receive VK102*/
  if (checkCommand(argc, argv, "-tcp_srv"))
  {
    g_flag_show_udp_send = false;
    boost::thread thread_tcp_server(tcpServerRun, argc, argv);
  }

  boost::thread thread_mqtt_send(mqttPubRun, argc, argv);

  msgs::StopInfoArray empty;
  RosModuleTraffic::publishReserve(TOPIC_RESERVE, empty);
  /*block main.*/
  while (true)
  {
    boost::this_thread::sleep(boost::posix_time::microseconds(1000000));
  }

  return 0;
}
