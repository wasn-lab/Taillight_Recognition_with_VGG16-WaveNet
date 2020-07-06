#include "all_header.h"
#include "GlobalVariable.h"
#include "UI/QtViewer.h"
#include "Transform_CUDA.h"
#include "LiDARStitchingAuto.h"
#include "CuboidFilter.h"
#include "NoiseFilter.h"
#include "extract_Indices.h"

//---------------------------- pointcloud
// no-filter PointCloud
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontLeft(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontRight(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontTop(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarAll(new pcl::PointCloud<pcl::PointXYZI>);

// Noise PointCloud
// pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontTop_Noise(new pcl::PointCloud<pcl::PointXYZI>);
// pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontLeft_Noise(new pcl::PointCloud<pcl::PointXYZI>);
// pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontRight_Noise(new pcl::PointCloud<pcl::PointXYZI>);
// pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarAll_Noise(new pcl::PointCloud<pcl::PointXYZI>);

// Focus PointCloud
// pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontTop_Focus(new pcl::PointCloud<pcl::PointXYZI>);
// pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontLeft_Focus(new pcl::PointCloud<pcl::PointXYZI>);
// pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontRight_Focus(new pcl::PointCloud<pcl::PointXYZI>);
// pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarAll_Focus(new pcl::PointCloud<pcl::PointXYZI>);

//---------------------------- Publisher
// no-filter
ros::Publisher pub_LidarFrontLeft;
ros::Publisher pub_LidarFrontRight;
ros::Publisher pub_LidarFrontTop;
ros::Publisher pub_LidarAll;

// noise pub
ros::Publisher pub_LidarFrontTop_Noise;
ros::Publisher pub_LidarFrontLeft_Noise;
ros::Publisher pub_LidarFrontRight_Noise;
ros::Publisher pub_LidarAll_Noise;

//----------------------------- Stitching
vector<double> LidarFrontLeft_Fine_Param;
vector<double> LidarFrontRight_Fine_Param;
vector<double> Zero_Param(6, 0.0);

LiDARStitchingAuto LSA;

//--------------------------- Global Variables
mutex syncLock;

StopWatch stopWatch;
bool debug_output = false;
bool use_filter = false;

bool heartBeat[5] = { false, false, false, false, false };  //{ FrontLeft, FrontRight, RearLeft, RearRight, FrontTop }
int heartBeat_times[5] = { 0, 0, 0, 0, 0 };
int lidarAll_pubFlag = 4;

void syncLock_callback();
void checkPubFlag(int lidarNum);
void lidarAll_Pub(int lidarNum);


//------------------------------ Callback
// lidars_callback -> synLock_callback
void cloud_cb_LidarFrontLeft(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  if (input_cloud->width * input_cloud->height > 100)
  {
    heartBeat[0] = true;
    // check data from hardware
    if (debug_output && (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) < 3600)
    {
      uint64_t diff_time = (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) * 1000;
      cout << "[Left->Gbr]: " << diff_time << "ms" << endl;
    }
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_tmp(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*input_cloud, *input_cloud_tmp);

    // check stitching
    if (GlobalVariable::STITCHING_MODE_NUM == 1)
    {
      syncLock_callback();
    }
    else
    {
      heartBeat[0] = false;
      
      // Transfrom
      *input_cloud_tmp = Transform_CUDA().compute<PointXYZI>(
      input_cloud_tmp, LidarFrontLeft_Fine_Param[0], LidarFrontLeft_Fine_Param[1], LidarFrontLeft_Fine_Param[2], LidarFrontLeft_Fine_Param[3], LidarFrontLeft_Fine_Param[4],
      LidarFrontLeft_Fine_Param[5]);
            
      // filter
      *input_cloud_tmp = CuboidFilter().hollow_removal_IO<PointXYZI>(input_cloud_tmp, -7.0, 1, -1.4, 1.4, -3.0, 0.5, -15.0, 50.0, -15.0, 15.0, -5.0, 0.01);
      *input_cloud_tmp = NoiseFilter().runRadiusOutlierRemoval<PointXYZI>(input_cloud_tmp, 0.22, 1);
      
      // assign
      *cloudPtr_LidarFrontLeft = *input_cloud_tmp;

      // publish
      cloudPtr_LidarFrontLeft->header.frame_id = "lidar";
      pub_LidarFrontLeft.publish(*cloudPtr_LidarFrontLeft);
    }  
  }
}

void cloud_cb_LidarFrontRight(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  if (input_cloud->width * input_cloud->height > 100)
  {
    heartBeat[0] = true;

    // check data from hardware
    if (debug_output && (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) < 3600)
    {
      uint64_t diff_time = (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) * 1000;
      cout << "[Right->Gbr]: " << diff_time << "ms" << endl;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_tmp(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*input_cloud, *input_cloud_tmp);

    // check stitching
    if (GlobalVariable::STITCHING_MODE_NUM == 1)
    {
      syncLock_callback();
    }
    else
    {
      heartBeat[0] = false;
      
      // Transfrom
      *input_cloud_tmp = Transform_CUDA().compute<PointXYZI>(
      input_cloud_tmp, LidarFrontRight_Fine_Param[0], LidarFrontRight_Fine_Param[1], LidarFrontRight_Fine_Param[2], LidarFrontRight_Fine_Param[3], LidarFrontRight_Fine_Param[4],
      LidarFrontRight_Fine_Param[5]);

      // filter
      *input_cloud_tmp = CuboidFilter().hollow_removal_IO<PointXYZI>(input_cloud_tmp, -7.0, 1, -1.4, 1.4, -3.0, 0.5, -15.0, 50.0, -15.0, 15.0, -5.0, 0.01);
      *input_cloud_tmp = NoiseFilter().runRadiusOutlierRemoval<PointXYZI>(input_cloud_tmp, 0.22, 1);
      
      // assign
      *cloudPtr_LidarFrontRight = *input_cloud_tmp;
      
      // publish
      cloudPtr_LidarFrontRight->header.frame_id = "lidar";
      pub_LidarFrontRight.publish(*cloudPtr_LidarFrontRight);
    }  
  }
}

void cloud_cb_LidarFrontTop(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  if (input_cloud->width * input_cloud->height > 100)
  {
    stopWatch.reset();
    heartBeat[0] = true;

    // check data from hardware
    if (debug_output && (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) < 3600)
    {
      uint64_t diff_time = (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) * 1000;
      cout << "[Top->Gbr]: " << diff_time << "ms" << endl;
    }
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_tmp(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*input_cloud, *input_cloud_tmp);

    // check stitching
    if (GlobalVariable::STITCHING_MODE_NUM == 1)
    {
      syncLock_callback();
    }
    else
    {
      heartBeat[0] = false;
      
      // Transfrom
      *input_cloud_tmp = Transform_CUDA().compute<PointXYZI>(input_cloud_tmp, 0, 0, 0, 0, 0.2, 0);
      
      // filter
      *input_cloud_tmp = CuboidFilter().hollow_removal_IO<PointXYZI>(input_cloud_tmp, -7.0, 1, -1.4, 1.4, -3.0, 0.5, -15.0, 50.0, -15.0, 15.0, -5.0, 0.01);
      *input_cloud_tmp = NoiseFilter().runRadiusOutlierRemoval<PointXYZI>(input_cloud_tmp, 0.22, 1);
      
      // assign
      *cloudPtr_LidarFrontTop = *input_cloud_tmp;

      // publish
      cloudPtr_LidarFrontTop->header.frame_id = "lidar";
      pub_LidarFrontTop.publish(*cloudPtr_LidarFrontTop);
    }  
  }
}


float return_MaxTimeDiff(float a, float b , float c)
{
  float diff_ab = a - b;
  float diff_ac = a - c;
 
  if (( diff_ab < 0 && diff_ac < 0 ) || ( diff_ab > 0 && diff_ac > 0) )
  {
    if (abs(diff_ab) > abs(diff_ac))
    {
      return diff_ab;
    }
    else
    {
      return diff_ac;
    }
  }
  else
  {
    return diff_ab + diff_ac;
  }
}


void lidarAll_Pub(int lidarNum)
{
  // check time sync
  // float Max_Time_Diff;

  // Max_Time_Diff = return_MaxTimeDiff( cloudPtr_LidarFrontTop->header.stamp.toSec(), 
  // cloudPtr_LidarFrontLeft->header.stamp.toSec(), cloudPtr_LidarFrontRight->header.stamp.toSec());
 
  // if (Max_Time_Diff > 0.05) 
  // {
  //   cout << "!-----------------------> Lidars is not Synchromized in  " << Max_Time_Diff << "s" << endl;
  // }  
   
  
  //------ combine pointcloud
  *cloudPtr_LidarAll += *cloudPtr_LidarFrontLeft;
  *cloudPtr_LidarAll += *cloudPtr_LidarFrontRight;
  *cloudPtr_LidarAll += *cloudPtr_LidarFrontTop;

  //------ assign header
  if (cloudPtr_LidarFrontTop->header.stamp != 0)
  {
    cloudPtr_LidarAll->header.stamp = cloudPtr_LidarFrontTop->header.stamp;
  }
  else
  {
    if (cloudPtr_LidarFrontLeft->header.stamp >= cloudPtr_LidarFrontRight->header.stamp)
    {
      cloudPtr_LidarAll->header.stamp = cloudPtr_LidarFrontLeft->header.stamp;
    }
    else
    {
      cloudPtr_LidarAll->header.stamp = cloudPtr_LidarFrontRight->header.stamp;
    }
  }
  uint64_t LidarAll_time;
  LidarAll_time = cloudPtr_LidarAll->header.stamp;
  cloudPtr_LidarAll->header.frame_id = "lidar";


  //------ pub LidarAll
  pub_LidarAll.publish(*cloudPtr_LidarAll);
  cloudPtr_LidarAll->clear();

  if (debug_output)
  {
    cout << "[Grabber]: " << stopWatch.getTimeSeconds() << 's' << endl;
  }

  //------ clear memory
  // if wall_time - ros_time !> 30 minutes, (not rosbag)
  // clear sensor pc data memory if delay 2sec.
  uint64_t now = ros::Time::now().toNSec() / 1000ull;  // microsec
  if (!((now - LidarAll_time) > 1000000 * 1800))
  {
    if ((now - cloudPtr_LidarFrontLeft->header.stamp) > 1000000 * 2)
    {
      cloudPtr_LidarFrontLeft->clear();
      cout << "---------------------> Front-Left Clear" << endl;
    };
    if ((now - cloudPtr_LidarFrontRight->header.stamp) > 1000000 * 2)
    {
      cloudPtr_LidarFrontRight->clear();
      cout << "---------------------> Front-Right Clear" << endl;
    };
    if ((now - cloudPtr_LidarFrontTop->header.stamp) > 1000000 * 2)
    {
      cloudPtr_LidarFrontTop->clear();
      cout << "---------------------> Top Clear" << endl;
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

void LidarAll_Publisher(int argc, char** argv)
{
  ros::Rate loop_rate(20);  // 80Hz
  while (ros::ok())
  {
    lidarAll_Pub(4);
    loop_rate.sleep();
  }
}


int main(int argc, char** argv)
{
  cout << "=============== Grabber Start ===============" << endl;

  ros::init(argc, argv, "lidars_grabber");
  ros::NodeHandle n;

  // check debug mode
  ros::param::get("/debug_output", debug_output);
  ros::param::get("/use_filter", use_filter);

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
  ros::Subscriber sub_LidarFrontLeft =
      n.subscribe<sensor_msgs::PointCloud2>("/LidarFrontLeft/Raw", 1, cloud_cb_LidarFrontLeft);
  ros::Subscriber sub_LidarFrontRight =
      n.subscribe<sensor_msgs::PointCloud2>("/LidarFrontRight/Raw", 1, cloud_cb_LidarFrontRight);
  ros::Subscriber sub_LidarFrontTop =
      n.subscribe<sensor_msgs::PointCloud2>("/LidarFrontTop/Raw", 1, cloud_cb_LidarFrontTop);


  // no-filter publisher
  pub_LidarFrontLeft = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarFrontLeft", 1);
  pub_LidarFrontRight = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarFrontRight", 1);
  pub_LidarFrontTop = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarFrontTop", 1);
  pub_LidarAll = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarAll", 1);

  thread TheadDetection_UI(UI, argc, argv);
  thread TheadDetection_Pub(LidarAll_Publisher, argc, argv);

  //----------------------------------- ros
  ros::AsyncSpinner spinner(4);
  spinner.start();

  TheadDetection_UI.join();
  TheadDetection_Pub.join();
  ros::waitForShutdown();

  

  cout << "=============== Grabber Stop ===============" << endl;
  return 0;
}
