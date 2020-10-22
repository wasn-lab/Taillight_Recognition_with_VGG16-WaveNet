#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float64.h>
#include <tf/tf.h>
#include <fstream>
#include <ros/package.h>
#include <cmath>
#include "std_msgs/Int32.h"
#include "msgs/Flag_Info.h"

ros::Publisher goal_publisher;
ros::Publisher checkpoint_publisher;

int ORGS = 0;
int route_choose = 1;

double seg_x[2000] = {};
double seg_y[2000] = {};
double seg_z[2000] = {};
double ori_x[2000] = {};
double ori_y[2000] = {};
double ori_z[2000] = {};
double ori_w[2000] = {};
int read_index = 0;

int Dspace_Flag_last[8] = {0,0,0,0,0,0,0,0};

template <int size_readtmp>
void read_txt(std::string fpname, double (&SEG_X)[size_readtmp],double (&SEG_Y)[size_readtmp],
double (&SEG_Z)[size_readtmp],double (&ORI_X)[size_readtmp],double (&ORI_Y)[size_readtmp],double (&ORI_Z)[size_readtmp],double (&ORI_W)[size_readtmp])
{
  std::string fname = fpname;

    std::ifstream fin;
    char line[300];
    memset( line, 0, sizeof(line));

    fin.open(fname.c_str(),std::ios::in);
    if(!fin) 
    {
        std::cout << "Fail to import txt" <<std::endl;
        exit(1);
    }

    while(fin.getline(line,sizeof(line),'\n')) 
    {
      std::string nmea_str(line);
      std::stringstream ss(nmea_str);
      std::string token;

      getline(ss,token, ',');
      SEG_X[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      SEG_Y[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      SEG_Z[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      ORI_X[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      ORI_Y[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      ORI_Z[read_index] = atof(token.c_str());
      getline(ss,token, ',');
      ORI_W[read_index] = atof(token.c_str());
      read_index += 1;
    }
}

void Ini_route_01_bytxt()
{
  std::string fpname = ros::package::getPath("mission_input");
  std::string fpname_s = fpname + "/data/ITRI_route_01.txt"; // full route

  read_txt(fpname_s, seg_x, seg_y, seg_z, ori_x, ori_y, ori_z, ori_w);
}

void Ini_route_02_bytxt()
{
  std::string fpname = ros::package::getPath("mission_input");
  std::string fpname_s = fpname + "/data/ITRI_route_02.txt"; // full route - pass 51

  read_txt(fpname_s, seg_x, seg_y, seg_z, ori_x, ori_y, ori_z, ori_w);
}

void Ini_route_03_bytxt()
{
  std::string fpname = ros::package::getPath("mission_input");
  std::string fpname_s = fpname + "/data/ITRI_route_03.txt"; // full route - pass 51 - end 14

  read_txt(fpname_s, seg_x, seg_y, seg_z, ori_x, ori_y, ori_z, ori_w);
}

void Ini_busstop_bytxt()
{
  std::string fpname = ros::package::getPath("mission_input");
  std::string fpname_s = fpname + "/data/ITRI_busstop.txt"; // full route

  read_txt(fpname_s, seg_x, seg_y, seg_z, ori_x, ori_y, ori_z, ori_w);
}

void get_goal_point()
{
  geometry_msgs::PoseStamped input_goal;

  input_goal.header.frame_id = "map";
  input_goal.header.stamp = ros::Time::now();

  input_goal.pose.position.x = seg_x[0];
  input_goal.pose.position.y = seg_y[0];
  input_goal.pose.position.z = seg_z[0];
  input_goal.pose.orientation.x = ori_x[0];
  input_goal.pose.orientation.y = ori_y[0];
  input_goal.pose.orientation.z = ori_z[0];
  input_goal.pose.orientation.w = ori_w[0];

  goal_publisher.publish(input_goal);
  std::cout << "Set offline goal point!" << std::endl;
  ros::Rate loop_rate(2);
  ros::spinOnce();
  loop_rate.sleep();
}

void get_checkpoint_point()
{
  geometry_msgs::PoseStamped input_checkpoint;
 
  for (int i=1;i<read_index;i++)
  {
    input_checkpoint.header.frame_id = "map";
    input_checkpoint.header.stamp = ros::Time::now();
    input_checkpoint.pose.position.x = seg_x[i];
    input_checkpoint.pose.position.y = seg_y[i];
    input_checkpoint.pose.position.z = seg_z[i];
    input_checkpoint.pose.orientation.x = ori_x[i];
    input_checkpoint.pose.orientation.y = ori_y[i];
    input_checkpoint.pose.orientation.z = ori_z[i];
    input_checkpoint.pose.orientation.w = ori_w[i];
    checkpoint_publisher.publish(input_checkpoint);
    std::cout << "Set offline checkpoint " << i << " !" << std::endl;
    ros::Rate loop_rate(2);
    ros::spinOnce();
    loop_rate.sleep();
  }
}

void get_realtime_goal_point(int i)
{
  geometry_msgs::PoseStamped input_goal;

  input_goal.header.frame_id = "map";
  input_goal.header.stamp = ros::Time::now();

  input_goal.pose.position.x = seg_x[i];
  input_goal.pose.position.y = seg_y[i];
  input_goal.pose.position.z = seg_z[i];
  input_goal.pose.orientation.x = ori_x[i];
  input_goal.pose.orientation.y = ori_y[i];
  input_goal.pose.orientation.z = ori_z[i];
  input_goal.pose.orientation.w = ori_w[i];

  goal_publisher.publish(input_goal);
  std::cout << "Set real time goal point!" << std::endl;
  ros::Rate loop_rate(2);
  loop_rate.sleep();
}

void get_realtime_checkpoint_point(int i)
{
  geometry_msgs::PoseStamped input_checkpoint;
 
  input_checkpoint.header.frame_id = "map";
  input_checkpoint.header.stamp = ros::Time::now();
  input_checkpoint.pose.position.x = seg_x[i];
  input_checkpoint.pose.position.y = seg_y[i];
  input_checkpoint.pose.position.z = seg_z[i];
  input_checkpoint.pose.orientation.x = ori_x[i];
  input_checkpoint.pose.orientation.y = ori_y[i];
  input_checkpoint.pose.orientation.z = ori_z[i];
  input_checkpoint.pose.orientation.w = ori_w[i];

  checkpoint_publisher.publish(input_checkpoint);
  std::cout << "Set real time checkpoint " << i << " !" << std::endl;
  ros::Rate loop_rate(2);
  loop_rate.sleep();
}

void offline_realtime_goal_setting()
{
  if (ORGS == 0)
  {
    if (route_choose == 1)
    {
      Ini_route_01_bytxt();
    }
    else if (route_choose == 2)
    {
      Ini_route_02_bytxt();
    }
    else
    {
      Ini_route_03_bytxt();
    }
    
    get_goal_point();
    get_checkpoint_point();
  }
  else
  {
    Ini_busstop_bytxt();
    ros::spin();
  } 
}

void busstopinfoCallback(const msgs::Flag_Info::ConstPtr& msg)
{
  if (ORGS == 1)
  {
    int sum = msg->Dspace_Flag01 + msg->Dspace_Flag02 + msg->Dspace_Flag03 +	msg->Dspace_Flag04 + msg->Dspace_Flag05 +	msg->Dspace_Flag06 + msg->Dspace_Flag07 +	msg->Dspace_Flag08;
    float Dspace_Flag[8] = {msg->Dspace_Flag01,msg->Dspace_Flag02,msg->Dspace_Flag03,msg->Dspace_Flag04,msg->Dspace_Flag05,msg->Dspace_Flag06,msg->Dspace_Flag07,msg->Dspace_Flag08};
    for (int i = 0; i < 8; i++)
    {
      if (Dspace_Flag[i] != Dspace_Flag_last[i])
      {
        std::cout << "Bus Stop num : " << sum << std::endl;
        double bustop_id[8] = {};
        int index = 0;
        for (int i = 0; i < 8; i++)
        {
          if (Dspace_Flag[i] == 1)
          {
            bustop_id[index] = i;
            index = index + 1;
          }
        }
        get_realtime_goal_point(bustop_id[sum-1]);
        for (int i = 0; i < sum-1; i++)
        {
          get_realtime_checkpoint_point(bustop_id[i]);
        }
        // int round_count = msg->PX2_Flag01;
        for (int j = 0; j < 8; j++)
        {
          Dspace_Flag_last[j] = Dspace_Flag[j];
        }
        break;
      }
    }
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "mission_input");
  ros::NodeHandle node;

  ros::param::get(ros::this_node::getName()+"/ORGS", ORGS);
  ros::param::get(ros::this_node::getName()+"/route_choose", route_choose);
  ros::Subscriber busstopinfo = node.subscribe("/BusStop/Info", 1, busstopinfoCallback);
  goal_publisher = node.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal", 10, true);
  checkpoint_publisher = node.advertise<geometry_msgs::PoseStamped>("/checkpoint", 10, true);

  offline_realtime_goal_setting();

  //ros::spin();
  return 0;
};
