#include "drivenet/campub.h"

// Subscriber: 8 cams
ros::Subscriber g_cam_obj_fc;
ros::Subscriber g_cam_obj_ft_f;
ros::Subscriber g_cam_obj_ft_c;

ros::Subscriber g_cam_obj_rf;
ros::Subscriber g_cam_obj_rb;

ros::Subscriber g_cam_obj_lf;
ros::Subscriber g_cam_obj_lb;

ros::Subscriber g_cam_obj_bt;

// Publisher: 1, represent all cams
ros::Publisher g_cam_obj_all;

std_msgs::Header g_header_all;

std::vector<msgs::DetectedObject> g_arr_cam_obj_fc;
std::vector<msgs::DetectedObject> g_arr_cam_obj_ft_f;
std::vector<msgs::DetectedObject> g_arr_cam_obj_ft_c;
std::vector<msgs::DetectedObject> g_arr_cam_obj_rf;
std::vector<msgs::DetectedObject> g_arr_cam_obj_rb;
std::vector<msgs::DetectedObject> g_arr_cam_obj_lf;
std::vector<msgs::DetectedObject> g_arr_cam_obj_lb;
std::vector<msgs::DetectedObject> g_arr_cam_obj_bt;

std::string g_cam_obj_fc_topic_name;
std::string g_cam_obj_ft_f_topic_name;
std::string g_cam_obj_ft_c_topic_name;
std::string g_cam_obj_rf_topic_name;
std::string g_cam_obj_rb_topic_name;
std::string g_cam_obj_lf_topic_name;
std::string g_cam_obj_lb_topic_name;
std::string g_cam_obj_bt_topic_name;

void callback_CamObjFC(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  g_header_all = DetectMsg->header;
  g_arr_cam_obj_fc = DetectMsg->objects;
}

void callback_CamObjFTf(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  g_arr_cam_obj_ft_f = DetectMsg->objects;
}

void callback_CamObjFTc(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  g_arr_cam_obj_ft_c = DetectMsg->objects;
}

void callback_CamObjRF(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  g_arr_cam_obj_rf = DetectMsg->objects;
}

void callback_CamObjRB(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  g_arr_cam_obj_rb = DetectMsg->objects;
}

void callback_CamObjLF(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  g_arr_cam_obj_lf = DetectMsg->objects;
}

void callback_CamObjLB(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  g_arr_cam_obj_lb = DetectMsg->objects;
}

void callback_CamObjBT(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  g_arr_cam_obj_bt = DetectMsg->objects;
}

void collectRepub()
{
  // arrCamObjAll = arrCamObjBT + arrCamObjLB;
  msgs::DetectedObjectArray arr_cam_obj_all;
  size_t all_size = g_arr_cam_obj_fc.size() + g_arr_cam_obj_ft_f.size() + g_arr_cam_obj_ft_c.size() +
                    g_arr_cam_obj_rf.size() + g_arr_cam_obj_rb.size() + g_arr_cam_obj_lf.size() +
                    g_arr_cam_obj_lb.size() + g_arr_cam_obj_bt.size();
  arr_cam_obj_all.objects.reserve(all_size);
  arr_cam_obj_all.header = g_header_all;

  for (size_t i = 0; i < g_arr_cam_obj_fc.size(); i++)
  {
    g_arr_cam_obj_fc[i].camInfo.id = camera::id::front_bottom_60;
    arr_cam_obj_all.objects.push_back(g_arr_cam_obj_fc[i]);
  }

  for (size_t i = 0; i < g_arr_cam_obj_ft_f.size(); i++)
  {
    g_arr_cam_obj_ft_f[i].camInfo.id = camera::id::front_top_far_30;
    arr_cam_obj_all.objects.push_back(g_arr_cam_obj_ft_f[i]);
  }

  for (size_t i = 0; i < g_arr_cam_obj_ft_c.size(); i++)
  {
    g_arr_cam_obj_ft_c[i].camInfo.id = camera::id::front_top_close_120;
    arr_cam_obj_all.objects.push_back(g_arr_cam_obj_ft_c[i]);
  }

  for (size_t i = 0; i < g_arr_cam_obj_rf.size(); i++)
  {
    g_arr_cam_obj_rf[i].camInfo.id = camera::id::right_front_60;
    arr_cam_obj_all.objects.push_back(g_arr_cam_obj_rf[i]);
  }

  for (size_t i = 0; i < g_arr_cam_obj_rb.size(); i++)
  {
    g_arr_cam_obj_rb[i].camInfo.id = camera::id::right_back_60;
    arr_cam_obj_all.objects.push_back(g_arr_cam_obj_rb[i]);
  }

  for (size_t i = 0; i < g_arr_cam_obj_lf.size(); i++)
  {
    g_arr_cam_obj_lf[i].camInfo.id = camera::id::left_front_60;
    arr_cam_obj_all.objects.push_back(g_arr_cam_obj_lf[i]);
  }

  for (size_t i = 0; i < g_arr_cam_obj_lb.size(); i++)
  {
    g_arr_cam_obj_lb[i].camInfo.id = camera::id::left_back_60;
    arr_cam_obj_all.objects.push_back(g_arr_cam_obj_lb[i]);
  }

  for (size_t i = 0; i < g_arr_cam_obj_bt.size(); i++)
  {
    g_arr_cam_obj_bt[i].camInfo.id = camera::id::back_top_120;
    arr_cam_obj_all.objects.push_back(g_arr_cam_obj_bt[i]);
  }

  ROS_INFO_STREAM("HEADER: " << arr_cam_obj_all.header);

  g_cam_obj_all.publish(arr_cam_obj_all);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "CamPub");
  ros::NodeHandle nh;

  // cam60_0_topicName = camera::topics[camera::id::right_60];
  // cam60_0 = nh.subscribe(cam60_0_topicName + std::string("/compressed"), 1, callback_60_0_decode);

  g_cam_obj_fc_topic_name = camera::topics_obj[camera::id::front_bottom_60];
  g_cam_obj_ft_f_topic_name = camera::topics_obj[camera::id::front_top_far_30];
  g_cam_obj_ft_c_topic_name = camera::topics_obj[camera::id::front_top_close_120];
  g_cam_obj_rf_topic_name = camera::topics_obj[camera::id::right_front_60];
  g_cam_obj_rb_topic_name = camera::topics_obj[camera::id::right_back_60];
  g_cam_obj_lf_topic_name = camera::topics_obj[camera::id::left_front_60];
  g_cam_obj_lb_topic_name = camera::topics_obj[camera::id::left_back_60];
  g_cam_obj_bt_topic_name = camera::topics_obj[camera::id::back_top_120];

  // Subscribe msgs
  g_cam_obj_fc = nh.subscribe(g_cam_obj_fc_topic_name, 1, callback_CamObjFC);
  g_cam_obj_ft_f = nh.subscribe(g_cam_obj_ft_f_topic_name, 1, callback_CamObjFTf);
  g_cam_obj_ft_c = nh.subscribe(g_cam_obj_ft_c_topic_name, 1, callback_CamObjFTc);
  g_cam_obj_rf = nh.subscribe(g_cam_obj_rf_topic_name, 1, callback_CamObjRF);
  g_cam_obj_rb = nh.subscribe(g_cam_obj_rb_topic_name, 1, callback_CamObjRB);
  g_cam_obj_lf = nh.subscribe(g_cam_obj_lf_topic_name, 1, callback_CamObjLF);
  g_cam_obj_lb = nh.subscribe(g_cam_obj_lb_topic_name, 1, callback_CamObjLB);
  g_cam_obj_bt = nh.subscribe(g_cam_obj_bt_topic_name, 1, callback_CamObjBT);

  g_cam_obj_all = nh.advertise<msgs::DetectedObjectArray>(camera::detect_result, 8);

  ros::Rate loop_rate(30);

  while (ros::ok())
  {
    collectRepub();
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
