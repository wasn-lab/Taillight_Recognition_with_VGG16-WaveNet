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

bool flag_show_udp_send = true;
bool event_queue_switch = true;

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
boost::mutex mutex_queue;
boost::mutex mutex_ros;
boost::mutex mutex_trafficLight;
boost::mutex mutex_event_1;
boost::mutex mutex_event_2;
boost::mutex mutex_mqtt;
boost::mutex mutex_sensor;
//boost::mutex mutex_serverStatus;

// ros queue
std::queue<std::string> q;
// adv heading queue
std::queue<std::string> obuQueue;

std::queue<std::string> vkQueue;
std::queue<std::string> vkStatusQueue;

std::queue<json> mqttGNSSQueue;
std::queue<json> mqttBSMQueue;
std::queue<json> mqttECUQueue;
std::queue<json> mqttIMUQueue;
std::queue<json> mqttSensorQueue;

std::queue<std::string> trafficLightQueue;
std::queue<json> eventQueue1;
std::queue<json> eventQueue2;

TcpServer server;

msgs::DetectedObjectArray detObjArray;
msgs::LidLLA gps;
msgs::VehInfo vehInfo;
json fps_json_ = { { "key", 0 } };
std::string VK102Response;
std::string mileJson;
std::string eventJson;
std::string statusJson;
msgs::RouteInfo route_info;
std::string board_list="00000000";
int routeID = 2000;

MqttClient mqttPub;
bool isMqttConnected = false;

long event_recv_count = 0;
long event_send_count = 0;

const static double PI = 3.14;
// can data
double data[10] = { 0 };

// traffic light buffer
char buffer[1024];

std::string PLATE = "ITRI-ADV";
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

struct pose
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
};

struct batteryInfo
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

unsigned int mode; //模式 自動/半自動/手動/鎖定
float emergency_exit; //緊急出口
  
pose current_gnss_pose;
ArriveStop cuttent_arrive_stop;
IMU imu;

std::string current_spat = "";

VehicelStatus vs;
batteryInfo battery;


void getCurrentPath(){
  char cwd[PATH_MAX];
  if (getcwd(cwd, sizeof(cwd)) != NULL) {
    printf("Current working dir: %s\n", cwd);
  } else {
    perror("getcwd() error");
  }
}

json genMqttGnssMsg();
json genMqttBmsMsg();
json genMqttECUMsg(ecu_type);
json genMqttIMUMsg();


/*=========================tools begin=========================*/
bool checkCommand(int argc, char** argv, std::string command)
{
  for (int i = 0; i < argc; i++)
  {
    if (command.compare(argv[i]) == 0)
    {
      return true;
    }
  }
  return false;
}

char* log_Time()
{
  struct tm* ptm;
  struct timeb stTimeb;
  static char szTime[24];

  ftime(&stTimeb);
  ptm = localtime(&stTimeb.time);
  sprintf(szTime, "%04d-%02d-%02d %02d:%02d:%02d.%03d", ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday,
          ptm->tm_hour, ptm->tm_min, ptm->tm_sec, stTimeb.millitm);
  szTime[23] = 0;
  return szTime;
}

std::time_t convertStrToTimeStamp(std::string time)
{
  std::tm t{};
  std::istringstream ss(time);

  ss >> std::get_time(&t, "%Y-%m-%d %H:%M:%S");
  if (ss.fail())
  {
    throw std::runtime_error{ "failed to parse time string" };
  }
  std::time_t time_stamp = mktime(&t);
  return time_stamp;
}

bool convertBoolean(int state)
{
  if(state == 0) return false;
  return true;
}
/*=========================tools end=========================*/

/*========================= ROS callbacks begin=========================*/

void callback_detObj(const msgs::DetectedObjectArray& input)
{
  mutex_ros.lock();
  detObjArray = input;
  mutex_ros.unlock();
}

void callback_gps(const msgs::LidLLA& input)
{
  std::cout << "callback gps " << std::endl;
  mutex_ros.lock();
  gps = input;
  mutex_ros.unlock();
}

void callback_veh(const msgs::VehInfo& input)
{
  mutex_ros.lock();
  vehInfo = input;
  mutex_ros.unlock();
}

void callback_gnss2local(const geometry_msgs::PoseStamped::ConstPtr& input)
{
  mutex_ros.lock();
  tf::Quaternion gnss_q(input->pose.orientation.x, input->pose.orientation.y, input->pose.orientation.z,
                        input->pose.orientation.w);
  tf::Matrix3x3 gnss_m(gnss_q);
  current_gnss_pose.x = input->pose.position.x;
  current_gnss_pose.y = input->pose.position.y;
  current_gnss_pose.z = input->pose.position.z;
  gnss_m.getRPY(current_gnss_pose.roll, current_gnss_pose.pitch, current_gnss_pose.yaw);
  mutex_ros.unlock();
}

void callback_fps(const std_msgs::String::ConstPtr& input)
{
  mutex_ros.lock();
  std::string jsonString = input->data.c_str();
  try
  {
    fps_json_ = json::parse(jsonString);
  }
  catch (std::exception& e)
  {
    std::cout << "callback_fps message: " << e.what() << std::endl;
    return;
  }
  mutex_ros.unlock();
}

void callbackBusStopInfo(const msgs::Flag_Info::ConstPtr& input)
{
  //std::cout << "<<<<<<<<<<<<<<<callbackBusStopInfo>>>>>>>>>>>>>>>" << std::endl;
  board_list = "";
  float stop[8];
  memset(stop, 0, sizeof(stop));
  mutex_ros.lock();
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
      stopids.push_back(i + routeID + 1);
      board_list += "1";
    }else{
      board_list += "0";
    }
  }
  json J2;
  for (size_t j = 0; j < stopids.size(); j++)
  {
    int id = stopids[j];
    json obj;
    obj["stopid"] = id;
    J2.push_back(obj);
  }
  json J3;
  J3["msgInfo"] = "Success";
  J3["msgCode"] = 200;
  json J1;
  J1["messageObj"] = J3;
  J1["type"] = "M8.2.VK102";
  J1["plate"] = PLATE;
  J1["status"] = 0;
  J1["route_id"] = routeID;
  if (stopids.size() == 0)
  {
    J1["bus_stops"] = json::array();
  }
  else
  {
    J1["bus_stops"] = J2;
  }

  VK102Response = J1.dump();
  mutex_ros.unlock();
}

void callbackNextStop(const msgs::Flag_Info::ConstPtr& input)
{
  mutex_ros.lock();
  cuttent_arrive_stop.id = routeID + (int)input->Dspace_Flag01;
  cuttent_arrive_stop.status = (int)input->Dspace_Flag02;
  //cuttent_arrive_stop.round = (int) input->PX2_Flag01;
  mutex_ros.unlock();
}

void callbackMileage(const std_msgs::String::ConstPtr& input)
{
  mutex_ros.lock();
  mileJson = input->data.c_str();
  std::cout << "mile info: " << mileJson << std::endl;

  mutex_ros.unlock();
}

void callbackRound(const std_msgs::Int32::ConstPtr& input)
{
  cuttent_arrive_stop.round = (int) input->data;
}

void callbackEvent(const std_msgs::String::ConstPtr& input)
{
  using namespace std;
  eventJson = input->data.c_str();
  json J0 = json::parse(eventJson);
  json J1;
  
  J1["type"] = "M8.2.VK003";
  J1["deviceid"] = PLATE;
  J1["lat"] = gps.lidar_Lat;
  J1["lng"] = gps.lidar_Lon;  
  J1["module"] = J0.at("module");
  J1["status"] = J0.at("status");
  J1["event_str"] = J0.at("event_str");
  J1["timestamp"] = J0.at("timestamp");
  
  
  if(event_queue_switch)
  {
    cout << " push to queue1 event: " << J1.dump() << endl;
    mutex_event_1.lock();
    eventQueue1.push(J1);
    mutex_event_1.unlock();
  }
  else
  {
    cout << " push to queue2 event: " << J1.dump() << endl;
    mutex_event_2.lock();
    eventQueue2.push(J1);
    mutex_event_2.unlock();
  }
  event_recv_count ++;
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
  imu.Gx = input->linear_acceleration.x;
  imu.Gy = input->linear_acceleration.y;
  imu.Gz = input->linear_acceleration.z;
  imu.Gyrox = input->angular_velocity.x;
  imu.Gyroy = input->angular_velocity.y;
  imu.Gyroz = input->angular_velocity.z;
}

void callbackBI(const msgs::BackendInfo::ConstPtr& input)
{
  vs.motor_temperature = input->motor_temperature; //馬達溫度
  vs.tire_pressure = input->tire_pressure; //胎壓
  vs.air_pressure = input->air_pressure; //氣壓
  vs.battery =  input->battery; //電量 %
  vs.steer = input->steer; //轉向
  vs.localization = input->localization; //定位
  vs.odometry = input->odometry; //里程
  vs.speed = input->speed; //車速 km/hr
  vs.rotating_speed = input->speed ; //轉速
  vs.bus_stop = input->bus_stop; //站點
  vs.vehicle_number = input->vehicle_number; //車號
  vs.gear = input->gear; //檔位
  vs.hand_brake = input->hand_brake; //手煞車
  vs.steering_wheel = input->steering_wheel; //方向盤
  vs.door = input->door; //車門
  vs.air_conditioner = input->air_conditioner; //空調
  vs.radar = input->radar; //radar
  vs.lidar = input->lidar;
  vs.camera = input->camera;
  vs.GPS = input->GPS;
  vs.headlight = input->headlight; //大燈 0關1開
  vs.wiper = input->wiper; //雨刷 0關 1開
  vs.indoor_light = input->indoor_light; //車內燈 0關 1開
  vs.gross_power = input->gross_power; //總電源 0關 1開
  vs.left_turn_light = input->left_turn_light; //左方向燈 0關 1開
  vs.right_turn_light = input->right_turn_light;//右方向燈 0關 1開
  vs.estop = input->estop; //E-Stop 0關 1開
  vs.ACC_state = input->ACC_state; //ACC電源狀態 0關 1開
  vs.time = input->time; //標準時間
  vs.driving_time = input->driving_time; //行駛時間
  vs.mileage = input->mileage; //行駛距離
  battery.gross_voltage = input->gross_voltage; //總電壓 V
  battery.gross_current = input->gross_current; //總電流 A
  battery.highest_voltage = input->highest_voltage; //最高電池電壓 0.01V
  battery.highest_number = input->highest_number; //最高電壓電池位置 電池編號
  battery.lowest_volage = input->lowest_volage; //最低電池電壓 0.01V
  battery.lowest_number = input->lowest_number; //最低電壓電池位置 電池編號
  battery.voltage_deviation = input->voltage_deviation; //高低電壓差 0.01V
  battery.highest_temp_location = input->highest_temp_location; //電池最高環境溫度位置 區域編號
  battery.highest_temperature = input->highest_temperature; //電池最高環境溫度
  mode = input->mode; //模式 自動/半自動/手動/鎖定
  emergency_exit = input->emergency_exit; //緊急出口
}


void callbackSersorStatus(const std_msgs::String::ConstPtr& input)
{
  using namespace std;
  mutex_sensor.lock();
  json J1 = json::parse(input->data.c_str());
  J1["vid"] = "dc5360f91e74";
  mqttSensorQueue.push(J1.dump());
  mutex_sensor.unlock();
}


/*========================= ROS callbacks end =========================*/

/*========================= json parsers begin =========================*/
std::string get_jsonmsg_can(const std::string& type, double* data)
{
  // std::string time_string = log_Time ();
  json J1;
  J1["type"] = type;
  J1["plate"] = PLATE;
  J1["deviceID"] = "00:00:00:00:00:01";
  J1["dt"] = log_Time();
  if (type == "M8.2.adv002")
  {
    J1["speed"] = data[0];
    J1["front_brake_pressure"] = data[1];
    J1["rear_brake_pressure"] = data[2];
    J1["steering_wheel_angle"] = data[3];
  }
  return J1.dump();
}

std::string get_jsonmsg_ros(const std::string& type)
{
  json J1;
  J1["type"] = type;
  J1["plate"] = PLATE;
  J1["deviceID"] = "00:00:00:00:00:01";
  J1["dt"] = log_Time();
  if (type == "M8.2.adv001")
  {
    J1["lat"] = gps.lidar_Lat;
    J1["lon"] = gps.lidar_Lon;
    J1["speed"] = -1;
    J1["bearing"] = -1;
    J1["turn_signal"] = -1;
  }
  else if (type == "M8.2.adv002")
  {
    J1["speed"] = vehInfo.ego_speed * 3.6;
    J1["front_brake_pressure"] = 0;
    J1["rear_brake_pressure"] = 0;
    J1["steering_wheel_angle"] = 0;
  }
  else if (type == "M8.2.adv003")
  {
    J1["sw_camera_signal"] = -1;
    J1["sw_lidar_signal"] = -1;
    J1["sw_radar_signal"] = -1;
    J1["slam"] = -1;
    J1["object_list"];
    int num = 0;
    for (size_t i = 0; i < detObjArray.objects.size(); i++)
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
      J2["object_adv_x"] = (detObjArray.objects[i].bPoint.p0.x + detObjArray.objects[i].bPoint.p7.x) / 2;
      J2["object_adv_y"] = (detObjArray.objects[i].bPoint.p0.y + detObjArray.objects[i].bPoint.p7.y) / 2;
      J2["object_type"] = detObjArray.objects[i].classId;
      J2["object_status"] = -1;
      J2["object_length"] = -1;
      J1["object_list"] += J2;
      num++;
    }
    J1["object_count"] = num;
  }
  return J1.dump();
}

std::string get_jsonmsg_to_obu(const std::string& type)
{
  json J1;
  if (type == "M8.2.adv009")
  {
    J1["type"] = type;
    J1["lat"] = std::to_string(gps.lidar_Lat);
    J1["lon"] = std::to_string(gps.lidar_Lon);
    J1["speed"] = std::to_string(data[0]);
    J1["bearing"] = std::to_string(current_gnss_pose.yaw * 180 / PI);
  }
  return J1.dump();
}

std::string get_jsonmsg_to_vk_server(const std::string& type)
{
  json J1;
  J1["type"] = type;
  J1["deviceid"] = PLATE; //PLATE;
  J1["receivetime"] = log_Time();
  if (type == "M8.2.VK001")
  {
    J1["motor"] = vs.motor_temperature; // 馬達溫度 //2.1;
    J1["tirepressure"] =  vs.tire_pressure; //胎壓 //0.0;
    J1["airpressure"] = vs.air_pressure; //氣壓 //0.0;
    J1["electricity"] = vs.battery; //電量//0.0;
    J1["steering"] = vs.steer; // 轉向 
    J1["bearing"] = current_gnss_pose.yaw * 180 / PI;
    J1["heading"] = 0.0;
    J1["milage"] =  vs.odometry; //行駛距離//0.0;
    J1["speed"] = data[0]; //vs.speed 車速 目前來源CAN
    J1["rotate"] = vs.rotating_speed; //轉速 //0.0;
    J1["gear"] = vs.gear; //檔位 //1;
    J1["handcuffs"] = convertBoolean(vs.hand_brake); //手煞車 //true;
    J1["Steeringwheel"] =data[3]; //方向盤 //0.0;
    J1["door"] = convertBoolean(vs.door); //車門 //true;
    J1["airconditioner"] = convertBoolean(vs.air_conditioner); //空調;
    J1["lat"] = gps.lidar_Lat; //vs.location 目前來源 lidar_lla
    J1["lng"] = gps.lidar_Lon; //vs.location 目前來源 lidar_lla
    J1["headlight"] = convertBoolean(vs.headlight); //車燈 //true;
    J1["wiper"] =  convertBoolean(vs.wiper); //雨刷//true;
    J1["Interiorlight"] = convertBoolean(vs.indoor_light); //車內燈//true;
    J1["mainswitch"] = convertBoolean(vs.gross_power); //總電源//true;
    J1["leftlight"] = convertBoolean(vs.left_turn_light); //左方向燈; //true
    J1["rightlight"] = convertBoolean(vs.right_turn_light); //右方向燈//true;
    J1["EStop"] = convertBoolean(vs.estop); // E-Stop//true;
    J1["ACCpower"] = convertBoolean(vs.ACC_state); //ACC 電源//true;
    J1["ArrivedStop"] = cuttent_arrive_stop.id; //目前來源 NextStop/Info
    J1["ArrivedStopStatus"] = cuttent_arrive_stop.status; // 目前來源NextStop/Info
    J1["round"] = cuttent_arrive_stop.round; //目前來源 BusStop/Round
    J1["route_id"] = routeID;  //預設2000
    J1["RouteMode"] = 2;
    J1["Gx"] = imu.Gx;   //   目前來源 imu_data_rad
    J1["Gy"] = imu.Gy;   //   目前來源 imu_data_rad
    J1["Gz"] = imu.Gz;   //   目前來源 imu_data_rad
    J1["Gyrox"] = imu.Gyrox; // 目前來源 imu_data_rad
    J1["Gyroy"] = imu.Gyroy; // 目前來源 imu_data_rad
    J1["Gyroz"] = imu.Gyroz; // 目前來源 imu_data_rad
    J1["accelerator"] = data[4]; //無rostopic 目前來源CAN
    J1["brake_pedal"] = data[5]; //無rostopic 目前來源CAN
    J1["distance"] = 0.0; //? 跟mileage有何不同？
    J1["mainvoltage"] = battery.gross_voltage; //總電壓//0.0;
    J1["maxvoltage"] = battery.highest_voltage; //最高電池電壓//0.0;
    J1["maxvbatteryposition"] =  battery.highest_number; //最高電壓電池編號//"5517XW";
    J1["minvoltage"] = battery.lowest_volage; //最低電池電壓//0.0;
    J1["pressurediff"] = battery.voltage_deviation; //高低電壓差//0.0;
    J1["maxtbatteryposition"] = battery.lowest_number; //最低電池電壓 0.01V"454FG"; 
    J1["maxtemperature"] = battery.highest_temperature; //電池最高環境溫度//0.0;
    J1["Signal"] = current_spat; //無資料
    J1["CMS"] = 1; //無資料
    J1["setting"] = mode; // 自動/半自動/手動/鎖定
    J1["board_list"] = board_list;
  }
  else if (type == "M8.2.VK002")
  {
    J1["motor"] = vs.motor_temperature; // 馬達溫度 //2.1;
    J1["tirepressure"] = vs.tire_pressure; //胎壓 //0.0;
    J1["airpressure"] = vs.air_pressure; //氣壓 //0.0;
    J1["electricity"] =  vs.battery; //電量//0.0;
    J1["steering"] = vs.steer; // 轉向 
    J1["bearing"] = current_gnss_pose.yaw * 180 / PI;
    J1["heading"] = 0.0;
    J1["milage"] = vs.odometry; //行駛距離//0.0;
    J1["speed"] = data[0]; //vs.speed 車速 目前來源CAN
    J1["rotate"] = vs.rotating_speed; //轉速 //0.0;
    J1["gear"] = vs.gear; //檔位 //1;
    J1["handcuffs"] = convertBoolean(vs.hand_brake); //手煞車 //true;
    J1["Steeringwheel"] = data[3]; //方向盤 //0.0;
    J1["door"] = convertBoolean(vs.door); //車門 //true;
    J1["airconditioner"] = convertBoolean(vs.air_conditioner); //空調;
    J1["lat"] = gps.lidar_Lat;  //vs.location 目前來源 lidar_lla
    J1["lng"] = gps.lidar_Lon;  //vs.location 目前來源 lidar_lla
    J1["headlight"] = convertBoolean(vs.headlight); //車燈 //true;
    J1["wiper"] = convertBoolean(vs.wiper); //雨刷//true;
    J1["Interiorlight"] = convertBoolean(vs.indoor_light); //車內燈//true;
    J1["mainswitch"] = convertBoolean(vs.gross_power); //總電源//true;
    J1["leftlight"] = convertBoolean(vs.left_turn_light); //左方向燈; //true
    J1["rightlight"] = convertBoolean(vs.right_turn_light); //右方向燈//true;
    J1["EStop"] = convertBoolean(vs.estop); // E-Stop//true;
    J1["ACCpower"] = convertBoolean(vs.ACC_state); //ACC 電源//true;
    J1["route_id"] = routeID; //default 2000
    J1["RouteMode"] = mode;
    J1["Gx"] = imu.Gx; //   目前來源 imu_data_rad
    J1["Gy"] = imu.Gy; //   目前來源 imu_data_rad
    J1["Gz"] = imu.Gz; //   目前來源 imu_data_rad
    J1["Gyrox"] = imu.Gyrox; //   目前來源 imu_data_rad
    J1["Gyroy"] = imu.Gyroy; //   目前來源 imu_data_rad
    J1["Gyroz"] = imu.Gyroz; //   目前來源 imu_data_rad
    J1["accelerator"] = data[4]; //無rostopic 目前來源CAN
    J1["brake_pedal"] = data[5]; //無rostopic 目前來源CAN
    J1["ArrivedStop"] = cuttent_arrive_stop.id; //目前來源 NextStop/Info
    J1["ArrivedStopStatus"] = cuttent_arrive_stop.status; //目前來源 NextStop/Info
    J1["round"] = cuttent_arrive_stop.round; //目前來源 BusStop/Round
    J1["Signal"] = current_spat; //無資料
    J1["CMS"] = 1; //無資料
    J1["setting"] = mode; 
    J1["EExit"] = emergency_exit; 
    J1["board_list"] = board_list;
  }else if (type == "M8.2.VK003"){
    J1["lat"] = gps.lidar_Lat;
    J1["lng"] = gps.lidar_Lon;
    json J0 = json::parse(eventJson);
    J1["module"] = J0.at("module");
    J1["status"] = J0.at("status");
    J1["event_str"] = J0.at("event_str");
    J1["timestamp"] = J0.at("timestamp");
  }else if (type == "M8.2.VK004")
  {
    for (int i = 0; i < FPS_KEY_LEN; i++)
    {
      std::string key = keys[i];
      float value = fps_json_.value(key, -1);
      J1[key] = value;
    }
  }
  else if (type == "M8.2.VK006")
  {
    // Roger 20200212 [ fix bug: resend the same json
    try
    {
      json J0 = json::parse(mileJson);
      J1["mileage_info"] = J0;
    }
    catch (std::exception& e)
    {
      //std::cout << "mileage: " << e.what() << std::endl;
    }
    
    mileJson = "";
    // Roger 20200212 ]
  }
  return J1.dump();
}
/*========================= json parsers end =========================*/

/*========================= thread runnables begin =========================*/
void mqtt_pubish(std::string msg)
{
  if(isMqttConnected){
      std::string topic = "vehicle/report/dc5360f91e74";
      std::cout << "publish "  << msg << std::endl;
      mqttPub.publish(topic, msg);
    }
}

void sendRun(int argc, char** argv)
{
  using namespace std;
  UdpClient UDP_Back_client;
  UdpClient UDP_OBU_client;
  UdpClient UDP_VK_client;
  UdpClient UDP_TABLET_client;
  UdpClient UDP_VK_FG_client;
  

  UDP_Back_client.initial(UDP_AWS_SRV_ADRR, UDP_AWS_SRV_PORT);
  UDP_OBU_client.initial(UDP_OBU_ADRR, UDP_OBU_PORT);
  UDP_VK_client.initial(UDP_VK_SRV_ADRR, UDP_VK_SRV_PORT);
  UDP_TABLET_client.initial("192.168.1.3", 9876);
  UDP_VK_FG_client.initial("140.134.128.42", 8888);
  

  // UDP_VK_client.initial("192.168.43.24", UDP_VK_SRV_PORT);
  while (true)
  {
    mutex_queue.lock();
    while (q.size() != 0)
    {
      UDP_Back_client.send_obj_to_server(q.front(), flag_show_udp_send);
      q.pop();
    }

    while (obuQueue.size() != 0)
    {
      UDP_OBU_client.send_obj_to_server(obuQueue.front(), flag_show_udp_send);
      obuQueue.pop();
    }

    while (vkQueue.size() != 0)
    {
      UDP_VK_client.send_obj_to_server(vkQueue.front(), flag_show_udp_send);
      //UDP_TABLET_client.send_obj_to_server(vkQueue.front(), flag_show_udp_send);
      vkQueue.pop();
    }
    
    while (vkStatusQueue.size() != 0)
    {
      UDP_VK_client.send_obj_to_server(vkStatusQueue.front(), true);
      UDP_VK_FG_client.send_obj_to_server(vkStatusQueue.front(), true);
      UDP_TABLET_client.send_obj_to_server(vkStatusQueue.front(), flag_show_udp_send);
      vkStatusQueue.pop();
    }
    mutex_queue.unlock();

    mutex_mqtt.lock();
    json J1;
    std::string states;
    J1["vid"] = "dc5360f91e74";
    json gnss_list = json::array();
    json bsm_list = json::array();
    json ecu_list = json::array();
    json imu_list = json::array();
    if(mqttGNSSQueue.size() !=0)
    {
      while(mqttGNSSQueue.size() != 0)
      {
        json gnss = mqttGNSSQueue.front();
        gnss_list.push_back(gnss);
        mqttGNSSQueue.pop();
      }
      J1["gnss"] = gnss_list;
    }

    if(mqttBSMQueue.size() != 0)
    {
      while(mqttBSMQueue.size() != 0){
        json bsm = mqttBSMQueue.front();
        bsm_list.push_back(bsm);
        mqttBSMQueue.pop();
      }
      J1["bms"] = bsm_list;
    }

    if(mqttECUQueue.size() != 0)
    {
      while(mqttECUQueue.size() != 0){
        json ecu = mqttECUQueue.front();
        ecu_list.push_back(ecu);
        mqttECUQueue.pop();
      }
      J1["ecu"] = ecu_list;
    }

    if(mqttIMUQueue.size() != 0)
    {
      while(mqttIMUQueue.size() != 0){
        json jimu = mqttIMUQueue.front();
        imu_list.push_back(jimu);
        mqttIMUQueue.pop();
      }
      J1["imu"] = imu_list;
    }
    if(mqttSensorQueue.size() != 0){
       mutex_sensor.lock();
       states = mqttSensorQueue.front();
       mqttSensorQueue.pop();
       mutex_sensor.unlock();
       mqtt_pubish(states);
    }

    mqtt_pubish(J1.dump());
    mutex_mqtt.unlock();


    if(event_queue_switch)
    {
      if(eventQueue1.size() != 0){
        mutex_event_1.lock();
        event_queue_switch = false;
      
        while (eventQueue1.size() != 0)
        {
          json j = eventQueue1.front();
          string jstr = j.dump();
          cout << "++++++++++++++++++++++++++++++send from q 1 " << jstr << endl;
          UDP_VK_client.send_obj_to_server(jstr, flag_show_udp_send);
          UDP_TABLET_client.send_obj_to_server(jstr, flag_show_udp_send);
          eventQueue1.pop();
        }

        mutex_event_1.unlock();
      }  
    }//if(event_queue_switch)
    else
    {
      if(eventQueue2.size() != 0){
        mutex_event_2.lock();
        event_queue_switch = true;
    
        while (eventQueue2.size() != 0)
        {
         
          json j = eventQueue2.front();
          string jstr = j.dump();
          cout << "+++++++++++++++++++++++++++++++send from q 2 " << jstr << endl;
          UDP_VK_client.send_obj_to_server(jstr, flag_show_udp_send);
          UDP_TABLET_client.send_obj_to_server(jstr, flag_show_udp_send);
          eventQueue2.pop();
        }

        mutex_event_2.unlock();
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
      int msg_id = receiver.receive(data);
      std::string type = get_msg_type(msg_id);
      std::string temp = get_jsonmsg_can(type, data);
      mutex_queue.lock();
      q.push(temp);
      mutex_queue.unlock();
    }
    receiver.closeSocket();
    boost::this_thread::sleep(boost::posix_time::microseconds(500000));
  }
}

void receiveUDPRun(int argc, char** argv)
{
  while (true)
  {
    UdpServer UDP_OBU_server(UDP_ADV_SRV_ADRR, UDP_ADV_SRV_PORT);
    //UdpServer UDP_OBU_server("192.168.43.204", UDP_ADV_SRV_PORT);
    int result = UDP_OBU_server.recv(buffer, sizeof(buffer));
    current_spat = "";
    if (result != -1)
    {
      mutex_trafficLight.lock();
      std::string tempStr(buffer);
      current_spat = tempStr;
      memset(buffer, 0, sizeof(buffer));
      trafficLightQueue.push(tempStr);
      mutex_trafficLight.unlock();
    }

    // 100 ms
    boost::this_thread::sleep(boost::posix_time::microseconds(UDP_SERVER_UPDATE_MICROSECONDS));
  }
}

void sendROSRun(int argc, char** argv)
{
  while (ros::ok())
  {
    mutex_trafficLight.lock();
    while (trafficLightQueue.size() != 0)
    {
      std::string trafficMsg = trafficLightQueue.front();
      trafficLightQueue.pop();
      msgs::Spat spat;
      json J0 = json::parse(trafficMsg);
      try
      {
        json J1 = J0.at("SPaT_MAP_Info");
        spat.lat = J1.at("Latitude");
        spat.lon = J1.at("Longitude");
        spat.spat_state = J1.at("Spat_state");
        spat.spat_sec = J1.at("Spat_sec");
        spat.signal_state = J1.at("Signal_state");
        spat.index = J1.at("Index");
      } 
      catch(std::exception& e)
      {
        std::cout << "parsing fail: " << e.what() << " "<<std::endl;
      }
      //send traffic light
      RosModuleTraffic::publishTraffic(TOPIC_TRAFFIC, spat);
    }
    mutex_trafficLight.unlock();
    
    //send route info
    RosModuleTraffic::publishRoute(TOPIC_ROUTE, route_info);

    boost::this_thread::sleep(boost::posix_time::microseconds(500000));
    ros::spinOnce();
  }
}

void receiveRosRun(int argc, char** argv)
{
  bool isBigBus = checkCommand(argc, argv, "-big");
  bool isNewMap = checkCommand(argc, argv, "-newMap");

  RosModuleTraffic::RegisterCallBack(callback_detObj, callback_gps, callback_veh, callback_gnss2local, callback_fps,
                                     callbackBusStopInfo, callbackMileage, callbackNextStop, callbackRound, callbackIMU, 
                                     callbackEvent, callbackBI, callbackSersorStatus, isNewMap);

  while (ros::ok())
  {
    mutex_ros.lock();
    std::string temp_adv001 = get_jsonmsg_ros("M8.2.adv001");
    mutex_queue.lock();
    q.push(temp_adv001);

    mutex_queue.unlock();

    std::string temp_adv003 = get_jsonmsg_ros("M8.2.adv003");
    mutex_queue.lock();
    q.push(temp_adv003);
    mutex_queue.unlock();

    std::string temp_adv009 = get_jsonmsg_to_obu("M8.2.adv009");
    mutex_queue.lock();
    obuQueue.push(temp_adv009);
    mutex_queue.unlock();

    if (isBigBus)
    {
      std::string temp_vk002 = get_jsonmsg_to_vk_server("M8.2.VK002");
      mutex_queue.lock();
      vkStatusQueue.push(temp_vk002);
      mutex_queue.unlock();
    }
    else
    {
      std::string temp_vk001 = get_jsonmsg_to_vk_server("M8.2.VK001");
      mutex_queue.lock();
      vkStatusQueue.push(temp_vk001);
      mutex_queue.unlock();
    }

    std::string temp_vk004 = get_jsonmsg_to_vk_server("M8.2.VK004");
    mutex_queue.lock();
    vkQueue.push(temp_vk004);
    mutex_queue.unlock();

    std::string temp_VK006 = get_jsonmsg_to_vk_server("M8.2.VK006");
    mutex_queue.lock();
    vkQueue.push(temp_VK006);
    mutex_queue.unlock();


    mutex_mqtt.lock();
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
    json imuobj = genMqttIMUMsg();


    mqttGNSSQueue.push(gnssobj);
    mqttBSMQueue.push(bsmobj);

    mqttECUQueue.push(ecu_acc_obj);
    mqttECUQueue.push(ecu_brk_obj);
    mqttECUQueue.push(ecu_speed_obj);
    mqttECUQueue.push(ecu_steer_obj);
    mqttECUQueue.push(ecu_geer_obj);
    mqttECUQueue.push(ecu_rpm_obj);
    mqttECUQueue.push(ecu_engineload_obj);
    mqttECUQueue.push(ecu_dtc_obj);

    mqttIMUQueue.push(imuobj);

    mutex_mqtt.unlock();


    mutex_ros.unlock();

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
    TCPClient TCP_VK_client;
    TCP_VK_client.initial(TCP_VK_SRV_ADRR, TCP_VK_SRV_PORT);
    // TCP_VK_client.initial("192.168.43.24", 8765);
    TCP_VK_client.connectServer();
    json J1;
    J1["type"] = "M8.2.VK005";
    J1["deviceid"] = "ITRI-ADV";
    std::string jsonString = J1.dump();
    const char* msg = jsonString.c_str();
    TCP_VK_client.sendRequest(msg, strlen(msg));
    TCP_VK_client.recvResponse(buffer_f, buff_size);
    std::string response(buffer_f);
    json J2;
    J2 = json::parse(response);
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
  json J1;
  json J2;

  J2["msgInfo"] = "Success";
  J2["msgCode"] = 200;
  J1["messageObj"] = J2;
  J1["status"] = status;
  return J1.dump();
}

std::string genErrorMsg(int code, std::string msg)
{
  json J1;
  json J2;

  J2["msgInfo"] = msg;
  J2["msgCode"] = code;
  J1["messageObj"] = J2;
  J1["type"] = "M8.2.VK102";
  J1["plate"] = PLATE;
  J1["status"] = 0;
  J1["route_id"] = routeID;
  J1["bus_stops"] = json::array();
  return J1.dump();
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
void VK102callback(std::string request)
{
  using namespace std;
  json J1;
  unsigned int in_round;
  unsigned int out_round;
  unsigned int in_stopid;
  unsigned int out_stopid;
  std::string type;

  // clear response
  VK102Response = "";

  // parsing
  try
  {
    J1 = json::parse(request);
  }
  catch (std::exception& e)
  {
    std::cout << "VK102callback message: " << e.what() << std::endl;
    // 400 bad request
    server.send_json(genErrorMsg(400, e.what()));
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
    server.send_json(genErrorMsg(400, e.what()));
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
  unsigned short retryCount = 0;
  while ( VK102Response.empty() && (retryCount < RESERVE_WAITING_TIMEOUT / REVERSE_SLEEP_TIME_MICROSECONDS ) )
  {
    retryCount ++;
    boost::this_thread::sleep(boost::posix_time::microseconds(REVERSE_SLEEP_TIME_MICROSECONDS));
  }
  
  /* response to server */
  if (VK102Response.empty())
  {
    server.send_json(genErrorMsg(201, "No data from /BusStop/Info."));
  }else {
    server.send_json(VK102Response);
  }
}


// response
void VK103callback(json reqJson)
{
  using namespace std;
  
  vector<unsigned int> stopids;
  
  // clear response
  VK102Response = "";
 
  cout << "VK103callback reqJson: " << reqJson.dump() << endl;

  // get data
  try
  {
    stopids = reqJson.at("stopid").get< vector<unsigned int> >();
  }
  catch (exception& e)
  {
    cout << "VK103callback message: " << e.what() << endl;
    server.send_json(genErrorMsg(400, e.what()));
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
  while ( VK102Response.empty() && (retryCount < RESERVE_WAITING_TIMEOUT / REVERSE_SLEEP_TIME_MICROSECONDS ) )
  {
    retryCount ++;
    std::cout << "retry: " << retryCount << std::endl;
    boost::this_thread::sleep(boost::posix_time::microseconds(REVERSE_SLEEP_TIME_MICROSECONDS));
  }
  
  /* response to server */
  if (VK102Response.empty())
  {
    server.send_json(genErrorMsg(201, "No data from /BusStop/Info."));
  }else {
    server.send_json(VK102Response);
  }
}

void VK104callback(json reqJson)
{
   using namespace std;
   
   cout << "VK104callback reqJson: " << reqJson.dump() << endl;
  
   string routePath = "";
   vector<unsigned int> stopids;
   

   // get data
   try
   {
     routeID = reqJson.at("routeid").get<int>();
     routePath = reqJson.at("routepath").get<string>(); 
     stopids = reqJson.at("stopid").get< vector<unsigned int> >();
   }
   catch (exception& e)
   {
     cout << "VK104callback message: " << e.what() << endl;
     server.send_json(genResMsg(0));
     return;
   }
 
   route_info.routeid = routeID;
   route_info.route_path = routePath;
   route_info.stops.clear();
   for (size_t i = 0 ; i < stopids.size(); i++)
   {
     msgs::StopInfo stop;
     stop.round = 1;
     stop.id = stopids[i];
     route_info.stops.push_back(stop);
   }

    /* check response from /BusStop/Info */ 
   unsigned short retryCount = 0;
   while ( VK102Response.empty() && (retryCount < RESERVE_WAITING_TIMEOUT / REVERSE_SLEEP_TIME_MICROSECONDS ) )
   {
     retryCount ++;
     boost::this_thread::sleep(boost::posix_time::microseconds(REVERSE_SLEEP_TIME_MICROSECONDS));
   }
  
   /* response to server */
   if (VK102Response.empty())
   {
     server.send_json(genResMsg(0));
   }else {
     server.send_json(genResMsg(1));
   }
}

//route api
void route(std::string request)
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
    server.send_json(genErrorMsg(400, e.what()));
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
    server.send_json(genErrorMsg(400, e.what()));
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
  server.initial(TCP_ADV_SRV_ADRR, TCP_ADV_SRV_PORT);
  //server.initial("192.168.43.204",8765);
  // server.initial("192.168.2.110",8765);
  // listening connection request
  int result = server.start_listening();
  
  if (result >= 0)
  {
    // accept and read request and handle request in VK102callback.
    try
    {
      server.wait_and_accept(route);
    }
    catch (std::exception& e)
    {
      server.send_json(genErrorMsg(408, "You should send request in 10 seconds after you connected to ADV."));
    }
  }
}

json genMqttGnssMsg()
{
  using namespace std::chrono;
  json gnss;
  uint64_t timestamp_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  //uint64_t source_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
  double lat = gps.lidar_Lat;
  double lon = gps.lidar_Lon;
  double alt = gps.lidar_Alt;
  gnss["coord"] = {lat, lon, alt};
  //gnss["speed"] = -1; remove speed
  gnss["heading"] = current_gnss_pose.yaw * 180 / PI;
  gnss["timestamp"] = timestamp_ms;
  gnss["source_time"] = timestamp_ms;
  return gnss;
}

json genMqttBmsMsg()
{
  using namespace std::chrono;
  uint64_t timestamp_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  json bsm;
  bsm["uid"] = PLATE;
  bsm["current"] = battery.gross_current;
  bsm["voltage"] = battery.gross_voltage;
  bsm["capacity"] =vs.battery;
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
      ecu["accelerator_pos"] = data[4];
      break;
    case ecu_type::brake_pos:
      ecu["brake_pos"] = data[5];
      break;
    case ecu_type::steering_wheel_angle:
      ecu["steering_wheel_angle"] = vs.steering_wheel;
      break;
    case ecu_type::speed:
      ecu["speed"] = data[0];
      break;
    case ecu_type::rpm:
      ecu["rpm"] =vs.rotating_speed;
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
      ecu["mileage"] = -1.0;
      break;
    case ecu_type::operation_speed:
      ecu["operation_speed"] = -1.0;
      ecu["maximum_speed"] = 35;
      break;
    case ecu_type::driving_mode:
      ecu["driving_mode"] = 1;
      break;
    case ecu_type::emergency_stop:
      ecu["emergency_stop"] = 1;
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
  jimu["uid"] = PLATE;
  //jimu["gyro_x"] = imu.Gyrox;
  //jimu["gyro_y"] = imu.Gyroy;
  //jimu["gyro_z"] = imu.Gyroz;
  jimu["gyro"] = {imu.Gyrox, imu.Gyroy, imu.Gyroz};
  jimu["roll_rate"] = current_gnss_pose.roll;
  jimu["pitch_rate"] = current_gnss_pose.pitch;
  jimu["yaw_rate"] = current_gnss_pose.yaw;
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

static void on_mqtt_connect(struct mosquitto* client, void* obj, int rc)
{
  std::string result;
  std::string topic = "vehicle/report/dc5360f91e74";
  switch (rc)
  {
    case 0:
      result = ": success";
      isMqttConnected = true;
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
  mqttPub.setOnConneclCallback(on_mqtt_connect);
  mqttPub.connect();
}

/*========================= thread runnables end =========================*/

int main(int argc, char** argv)
{
  using namespace std;
  RosModuleTraffic::Initial(argc, argv);
  PLATE = RosModuleTraffic::getPlate();
  //RosModuleTraffic::advertisePublisher();
  /*Start thread to receive data from can bus.*/
  if (!checkCommand(argc, argv, "-no_can"))
  {
    boost::thread ThreadCanReceive(receiveCanRun, argc, argv);
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
  boost::thread ThreadRosReceive(receiveRosRun, argc, argv);

  /*start thread for UDP client :
    1. AWS backend: API: adv001 adv003
    2. VK(.120) backenda: API: VK001 VK002 VK004
    3. OBU. API:adv009
  */
  boost::thread ThreadSend(sendRun, argc, argv);

  /*Start thread for UDP server to receive traffic light infomation from OBU. */
  if (checkCommand(argc, argv, "-udp_srv"))
  {
    flag_show_udp_send = false;
    boost::thread ThreadUDPreceive(receiveUDPRun, argc, argv);
  }
  /*Start thread to publish ROS topics:
    1. /traffic : traffic light from OBU to GUI.
    2. /serverStatus : server status for ADV_op/sys_ready.
    3. /reserve/request: Reserve request from back end to Control team.
  */
  boost::thread ThreadROSSend(sendROSRun, argc, argv);

  /*Strart thread for TCP client: Send VK005 to get server status.*/
  boost::thread ThreadGetServerStatus(getServerStatusRun, argc, argv);

  /*Startr thread for TCP server: Receive VK102*/
  if (checkCommand(argc, argv, "-tcp_srv"))
  {
    flag_show_udp_send = false;
    boost::thread ThreadTCPServer(tcpServerRun, argc, argv);
  }

  boost::thread ThreadMQTTSend(mqttPubRun, argc, argv);

  msgs::StopInfoArray empty;
  RosModuleTraffic::publishReserve(TOPIC_RESERVE, empty);
  /*block main.*/
  while (true)
  {
    boost::this_thread::sleep(boost::posix_time::microseconds(1000000));
  }

  return 0;
}
