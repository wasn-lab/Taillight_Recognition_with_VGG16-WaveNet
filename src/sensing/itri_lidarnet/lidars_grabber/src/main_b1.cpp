#include "all_header.h"
#include "GlobalVariable.h"
#include "UI/QtViewer.h"
#include "Transform_CUDA.h"
#include "LiDARStitchingAuto.h"
#include "CuboidFilter.h"

pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontLeft(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontRight(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarRearLeft(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarRearRight(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontTop(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidAll(new pcl::PointCloud<pcl::PointXYZI>);

ros::Publisher pub_LidarFrontLeft;
ros::Publisher pub_LidarFrontRight;
ros::Publisher pub_LidarRearLeft;
ros::Publisher pub_LidarRearRight;
ros::Publisher pub_LidarFrontTop;
ros::Publisher pub_LidAll;

vector<double> LidarFrontLeft_Fine_Param;
vector<double> LidarFrontRight_Fine_Param;
vector<double> LidarRearLeft_Fine_Param;
vector<double> LidarRearRight_Fine_Param;
vector<double> Zero_Param(6, 0.0);

LiDARStitchingAuto LSA;

mutex syncLock;

StopWatch stopWatch;

bool debug_output = false;

bool heartBeat[5] = { false, false, false, false, false };  //{ FrontLeft, FrontRight, RearLeft, RearRight, FrontTop }
int heartBeat_times[5] = { 0, 0, 0, 0, 0 };
int lidarAll_pubFlag = 4;

void syncLock_callback();
void checkPubFlag(int lidarNum);
void lidarAll_Pub(int lidarNum);



// lidars_callback -> synLock_callback
void cloud_cb_LidarFrontLeft(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  heartBeat[0] = true;
  pcl::fromROSMsg(*input_cloud, *cloudPtr_LidarFrontLeft);
  syncLock_callback();
}
void cloud_cb_LidarFrontRight(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  heartBeat[1] = true;
  pcl::fromROSMsg(*input_cloud, *cloudPtr_LidarFrontRight);
  syncLock_callback();
}
void cloud_cb_LidarFrontTop(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  heartBeat[4] = true;
  pcl::fromROSMsg(*input_cloud, *cloudPtr_LidarFrontTop);

  if (debug_output) 
  {
    ros::Time rosTime;
    pcl_conversions::fromPCL(cloudPtr_LidarFrontTop->header.stamp, rosTime);
    cout << "[Top->Gbr]: " << (ros::Time::now() - rosTime).toSec()*1000 << "ms" << endl;
    stopWatch.reset();
  }
  syncLock_callback();
}

void lidarAll_Pub(int lidarNum)
{
  switch (lidarNum)
  {
    case 0:
      cloudPtr_LidAll->header.seq = cloudPtr_LidarFrontLeft->header.seq;
      cloudPtr_LidAll->header.frame_id = cloudPtr_LidarFrontLeft->header.frame_id;
      break;
    case 1:
      cloudPtr_LidAll->header.seq = cloudPtr_LidarFrontRight->header.seq;
      cloudPtr_LidAll->header.frame_id = cloudPtr_LidarFrontRight->header.frame_id;
      break;
    case 4:
      cloudPtr_LidAll->header.seq = cloudPtr_LidarFrontTop->header.seq;
      cloudPtr_LidAll->header.frame_id = cloudPtr_LidarFrontTop->header.frame_id;
      break;
    default:
      cloudPtr_LidAll->header.seq = cloudPtr_LidarFrontTop->header.seq;
      cloudPtr_LidAll->header.frame_id = cloudPtr_LidarFrontTop->header.frame_id;
      break;
  }

  *cloudPtr_LidAll = *cloudPtr_LidarFrontLeft;
  *cloudPtr_LidAll += *cloudPtr_LidarFrontRight;
  *cloudPtr_LidAll += *cloudPtr_LidarRearLeft;
  *cloudPtr_LidAll += *cloudPtr_LidarRearRight;
  *cloudPtr_LidAll += *cloudPtr_LidarFrontTop;

  uint64_t avg_time;
  int n = 0;
  if (cloudPtr_LidarFrontLeft->header.stamp != 0)
    n += 1;
  if (cloudPtr_LidarFrontRight->header.stamp != 0)
    n += 1;
  if (cloudPtr_LidarFrontTop->header.stamp != 0)
    n += 1;

  if (n != 0)
  {
    avg_time = (cloudPtr_LidarFrontLeft->header.stamp + cloudPtr_LidarFrontRight->header.stamp + cloudPtr_LidarFrontTop->header.stamp) /  n;
  }
  else
  {
    avg_time = 0;
  }

  //cloudPtr_LidAll->header.stamp = cloudPtr_LidarFrontTop->header.stamp;
  
  // LidarALL time is publish time not Top time
  pcl_conversions::toPCL(ros::Time::now(), cloudPtr_LidAll->header.stamp);

  pub_LidAll.publish(*cloudPtr_LidAll);
  cloudPtr_LidAll->clear();
  
  if (debug_output)
  { 
      cout << "[Grabber]: " << stopWatch.getTimeSeconds() << "s" << endl;
  }

  // if wall_time - ros_time !> 30 minutes, (not rosbag)
  // clear sensor pc data memory if delay 3sec.
  uint64_t now = ros::Time::now().toNSec() / 1000ull;  // microsec
  if (!((now - avg_time) > 1000000 * 1800))
  {
    if ((now - cloudPtr_LidarFrontLeft->header.stamp) > 1000000 * 3)
    {
      cloudPtr_LidarFrontLeft->clear();
      cout << "Front-Left Clear" << endl;
    };
    if ((now - cloudPtr_LidarFrontRight->header.stamp) > 1000000 * 3)
    {
      cloudPtr_LidarFrontRight->clear();
      cout << "Front-Right Clear" << endl;
    };
    if ((now - cloudPtr_LidarFrontTop->header.stamp) > 1000000 * 3)
    {
      cloudPtr_LidarFrontTop->clear();
      cout << "Top Clear" << endl;
    };
  }
}

void checkPubFlag(int lidarNum)
{
  if (lidarAll_pubFlag == lidarNum)
  {
    // cout << "[PubFlag]: " << lidarNum << endl;
    lidarAll_Pub(lidarNum);
    heartBeat_times[0] = 0;
    heartBeat_times[1] = 0;
    heartBeat_times[4] = 0;
  }
  else if (lidarAll_pubFlag != lidarNum && heartBeat_times[lidarNum] > 3)
  {
    lidarAll_pubFlag = lidarNum;
    cout << "[PubFlag Change!]: " << lidarNum << endl;
    lidarAll_Pub(lidarNum);
    heartBeat_times[0] = 0;
    heartBeat_times[1] = 0;
    heartBeat_times[4] = 0;
  }
  else
  {
    heartBeat_times[lidarNum] += 1;
  }
}

void syncLock_callback()
{
  syncLock.lock();
    
  if (GlobalVariable::STITCHING_MODE_NUM == 1)
  {
    if (heartBeat[0] == true)
    {
      heartBeat[0] = false;

      if (GlobalVariable::FrontLeft_FineTune_Trigger == true)
      {
        Eigen::Matrix4f final_transform_tmp;

        GlobalVariable::FrontLeft_FineTune_Trigger = false;
        LSA.setInitTransform(GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1], GlobalVariable::UI_PARA[2],
                             GlobalVariable::UI_PARA[3], GlobalVariable::UI_PARA[4], GlobalVariable::UI_PARA[5]);
        LSA.updateEstimation(cloudPtr_LidarFrontLeft, cloudPtr_LidarFrontTop);  // src, base
        LSA.getFinalTransform(final_transform_tmp);

        cout << final_transform_tmp << endl;

        Eigen::Matrix3f m;
        m(0, 0) = final_transform_tmp(0, 0);
        m(0, 1) = final_transform_tmp(0, 1);
        m(0, 2) = final_transform_tmp(0, 2);
        m(1, 0) = final_transform_tmp(1, 0);
        m(1, 1) = final_transform_tmp(1, 1);
        m(1, 2) = final_transform_tmp(1, 2);
        m(2, 0) = final_transform_tmp(2, 0);
        m(2, 1) = final_transform_tmp(2, 1);
        m(2, 2) = final_transform_tmp(2, 2);

        Eigen::Vector3f ea = m.eulerAngles(0, 1, 2);
        cout << "to Euler angles:" << endl;
        cout << ea << endl;

        // write to GlobalVariable::UI_PARA[0~6]
        GlobalVariable::UI_PARA[0] = final_transform_tmp(0, 3);
        GlobalVariable::UI_PARA[1] = final_transform_tmp(1, 3);
        GlobalVariable::UI_PARA[2] = final_transform_tmp(2, 3);
        GlobalVariable::UI_PARA[3] = ea(0);
        GlobalVariable::UI_PARA[4] = ea(1);
        GlobalVariable::UI_PARA[5] = ea(2);
      }

      *cloudPtr_LidarFrontLeft = Transform_CUDA().compute<PointXYZI>(
          cloudPtr_LidarFrontLeft, GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1], GlobalVariable::UI_PARA[2],
          GlobalVariable::UI_PARA[3], GlobalVariable::UI_PARA[4], GlobalVariable::UI_PARA[5]);
      cloudPtr_LidarFrontLeft->header.frame_id = "lidar";
      pub_LidarFrontLeft.publish(*cloudPtr_LidarFrontLeft);
      checkPubFlag(0);
    }

    if (heartBeat[1] == true)
    {
      heartBeat[1] = false;

      if (GlobalVariable::FrontRight_FineTune_Trigger == true)
      {
        Eigen::Matrix4f final_transform_tmp;

        GlobalVariable::FrontRight_FineTune_Trigger = false;
        LSA.setInitTransform(GlobalVariable::UI_PARA[6], GlobalVariable::UI_PARA[7], GlobalVariable::UI_PARA[8],
                             GlobalVariable::UI_PARA[9], GlobalVariable::UI_PARA[10], GlobalVariable::UI_PARA[11]);
        LSA.updateEstimation(cloudPtr_LidarFrontRight, cloudPtr_LidarFrontTop);  // src, base
        LSA.getFinalTransform(final_transform_tmp);

        cout << final_transform_tmp << endl;

        Eigen::Matrix3f m;
        m(0, 0) = final_transform_tmp(0, 0);
        m(0, 1) = final_transform_tmp(0, 1);
        m(0, 2) = final_transform_tmp(0, 2);
        m(1, 0) = final_transform_tmp(1, 0);
        m(1, 1) = final_transform_tmp(1, 1);
        m(1, 2) = final_transform_tmp(1, 2);
        m(2, 0) = final_transform_tmp(2, 0);
        m(2, 1) = final_transform_tmp(2, 1);
        m(2, 2) = final_transform_tmp(2, 2);

        Eigen::Vector3f ea = m.eulerAngles(0, 1, 2);
        cout << "to Euler angles:" << endl;
        cout << ea << endl;

        // write to GlobalVariable::UI_PARA[0~6]
        GlobalVariable::UI_PARA[6] = final_transform_tmp(0, 3);
        GlobalVariable::UI_PARA[7] = final_transform_tmp(1, 3);
        GlobalVariable::UI_PARA[8] = final_transform_tmp(2, 3);
        GlobalVariable::UI_PARA[9] = ea(0);
        GlobalVariable::UI_PARA[10] = ea(1);
        GlobalVariable::UI_PARA[11] = ea(2);
      }

      *cloudPtr_LidarFrontRight = Transform_CUDA().compute<PointXYZI>(
          cloudPtr_LidarFrontRight, GlobalVariable::UI_PARA[6], GlobalVariable::UI_PARA[7], GlobalVariable::UI_PARA[8],
          GlobalVariable::UI_PARA[9], GlobalVariable::UI_PARA[10], GlobalVariable::UI_PARA[11]);

      cloudPtr_LidarFrontRight->header.frame_id = "lidar";
      pub_LidarFrontRight.publish(*cloudPtr_LidarFrontRight);
      checkPubFlag(1);
    }

    // LidarFrontTop does not need to compute
    if (heartBeat[4] == true)
    {
      heartBeat[4] = false;

      *cloudPtr_LidarFrontTop = Transform_CUDA().compute<PointXYZI>(cloudPtr_LidarFrontTop, 0, 0, 0, 0, 0.2, 0);

      cloudPtr_LidarFrontTop->header.frame_id = "lidar";

      pub_LidarFrontTop.publish(*cloudPtr_LidarFrontTop);
      checkPubFlag(4);
    }
  }
  //========================== ELSE AREA ==========================//
  else
  {
    if (heartBeat[0] == true)
    {
      heartBeat[0] = false;
      *cloudPtr_LidarFrontLeft = Transform_CUDA().compute<PointXYZI>(
          cloudPtr_LidarFrontLeft, LidarFrontLeft_Fine_Param[0], LidarFrontLeft_Fine_Param[1],
          LidarFrontLeft_Fine_Param[2], LidarFrontLeft_Fine_Param[3], LidarFrontLeft_Fine_Param[4],
          LidarFrontLeft_Fine_Param[5]);
      cloudPtr_LidarFrontLeft->header.frame_id = "lidar";
      pub_LidarFrontLeft.publish(*cloudPtr_LidarFrontLeft);
      checkPubFlag(0);
    }
    if (heartBeat[1] == true)
    {
      heartBeat[1] = false;
      *cloudPtr_LidarFrontRight = Transform_CUDA().compute<PointXYZI>(
          cloudPtr_LidarFrontRight, LidarFrontRight_Fine_Param[0], LidarFrontRight_Fine_Param[1],
          LidarFrontRight_Fine_Param[2], LidarFrontRight_Fine_Param[3], LidarFrontRight_Fine_Param[4],
          LidarFrontRight_Fine_Param[5]);
      cloudPtr_LidarFrontRight->header.frame_id = "lidar";
      pub_LidarFrontRight.publish(*cloudPtr_LidarFrontRight);
      checkPubFlag(1);
    }
    if (heartBeat[4] == true)
    {
      heartBeat[4] = false;
      *cloudPtr_LidarFrontTop = Transform_CUDA().compute<PointXYZI>(cloudPtr_LidarFrontTop, 0, 0, 0, 0, 0.2, 0);
      *cloudPtr_LidarFrontTop = CuboidFilter().hollow_removal<PointXYZI>(cloudPtr_LidarFrontTop, -20, -2, -2, 2, -5, 0);
      cloudPtr_LidarFrontTop->header.frame_id = "lidar";
      pub_LidarFrontTop.publish(*cloudPtr_LidarFrontTop);
      checkPubFlag(4);
    }
  }

  syncLock.unlock();
}

void UI(int argc, char** argv)
{
  if (pcl::console::find_switch(argc, argv, "-D"))
  {
    QApplication a(argc, argv);
    QtViewer w;
    w.show();
    a.exec();
  }
}

int main(int argc, char** argv)
{
  cout << "=============== Grabber Start ===============" << endl;

  ros::init(argc, argv, "lidars_grabber");
  ros::NodeHandle n;
  
  // check debug mode
  ros::param::get("/debug_output", debug_output);

  // check stitching mode
  if (pcl::console::find_switch(argc, argv, "-D"))
  {
    GlobalVariable::STITCHING_MODE_NUM = 1;
    for (int i = 0; i < 30; i++)
    {
      GlobalVariable::UI_PARA[i] = 0.0000000;
    };
    for (int i = 0; i < 30; i++)
    {
      GlobalVariable::UI_PARA_BK[i] = 0.0000000;
    };
  }

  // check stitching parameter
  if (!ros::param::has("/LidarFrontRight_Fine_Param"))
  {
    n.setParam("LidarFrontLeft_Fine_Param", Zero_Param);
    n.setParam("LidarFrontRight_Fine_Param", Zero_Param);

    cout << "NO STITCHING PARAMETER INPUT!" << endl;
    cout << "Now is using [0,0,0,0,0,0] as stitching parameter!" << endl;
  }
  else
  {
    n.param("/LidarFrontLeft_Fine_Param", LidarFrontLeft_Fine_Param, vector<double>());
    n.param("/LidarFrontRight_Fine_Param", LidarFrontRight_Fine_Param, vector<double>());
  

    cout << "STITCHING PARAMETER FIND!" << endl;
  }

  // subscriber
  ros::Subscriber sub_LidarFrontLeft  = n.subscribe<sensor_msgs::PointCloud2>("/LidarFrontLeft/Raw", 1, cloud_cb_LidarFrontLeft);
  ros::Subscriber sub_LidarFrontRight = n.subscribe<sensor_msgs::PointCloud2>("/LidarFrontRight/Raw", 1, cloud_cb_LidarFrontRight);
  ros::Subscriber sub_LidarFrontTop   =  n.subscribe<sensor_msgs::PointCloud2>("/LidarFrontTop/Raw", 1, cloud_cb_LidarFrontTop);

  // publisher
  pub_LidarFrontLeft = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarFrontLeft", 1);
  pub_LidarFrontRight = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarFrontRight", 1);
  pub_LidarFrontTop = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarFrontTop", 1);
  pub_LidAll = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarAll", 1);

  thread TheadDetection(UI, argc, argv);
  
  //ros
  ros::MultiThreadedSpinner s(3);
  ros::spin(s);

  // ros::Rate loop_rate(80);  // 80Hz
  // while (ros::ok())
  // {
  //   ros::spinOnce();
  //   loop_rate.sleep();
  // }

  cout << "=============== Grabber Stop ===============" << endl;
  return 0;
}
