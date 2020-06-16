#include "all_header.h"
#include "GlobalVariable.h"
#include "UI/QtViewer.h"
#include "Transform_CUDA.h"
#include "LiDARStitchingAuto.h"
#include "RosModuleHINO.hpp"

#define CHECKTIMES 20

pcl::PointCloud<pcl::PointXYZI>::Ptr cloudLeft(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudRight(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudFront(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudTop(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudLidarAll(new pcl::PointCloud<pcl::PointXYZI>);

LiDARStitchingAuto LSA;

mutex syncLock;
size_t heartBeat[4] = { 0 };  // left, right, front, top

void callback_LidarLeft(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  syncLock.lock();
  heartBeat[0] = 0;
  pcl::fromROSMsg(*input_cloud, *cloudLeft);
  syncLock.unlock();
}

void callback_LidarRight(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  syncLock.lock();
  heartBeat[1] = 0;
  pcl::fromROSMsg(*input_cloud, *cloudRight);
  syncLock.unlock();
}

void callback_LidarFront(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  syncLock.lock();
  heartBeat[2] = 0;
  pcl::fromROSMsg(*input_cloud, *cloudFront);
  syncLock.unlock();
}

void callback_LidarTop(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  syncLock.lock();
  heartBeat[3] = 0;
  pcl::fromROSMsg(*input_cloud, *cloudTop);
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
  RosModuleHINO::initial("lidars_grabber", argc, argv);
  cout << "[" << ros::this_node::getName() << "] "
       << "----------------------------startup" << endl;
  RosModuleHINO::RegisterCallBackLidarRaw(callback_LidarFront, callback_LidarLeft, callback_LidarRight,
                                          callback_LidarTop);

  vector<double> LidarLeft_Fine_Param;
  vector<double> LidarRight_Fine_Param;
  vector<double> LidarFront_Fine_Param;

  ros::NodeHandle n;

  if (ros::param::has("/LidarLeft_Fine_Param"))
  {
    cout << "STITCHING PARAMETER FIND!" << endl;

    n.param("/LidarLeft_Fine_Param", LidarLeft_Fine_Param, vector<double>());
    n.param("/LidarRight_Fine_Param", LidarRight_Fine_Param, vector<double>());
    n.param("/LidarFront_Fine_Param", LidarFront_Fine_Param, vector<double>());
  }
  else
  {
    cout << "NO STITCHING PARAMETER INPUT! Using [0,0,0,0,0,0] as stitching parameter" << endl;

    vector<double> Zero_Param(6, 0.0);
    n.setParam("LidarLeft_Fine_Param", Zero_Param);
    n.setParam("LidarRight_Fine_Param", Zero_Param);
    n.setParam("LidarFront_Fine_Param", Zero_Param);
  }

  if (pcl::console::find_switch(argc, argv, "-D"))
  {
    GlobalVariable::STITCHING_MODE_NUM = 1;
  }

  thread TheadDetection(UI, argc, argv);

  StopWatch stopWatch;

  ros::Rate loop_rate(20);
  while (ros::ok())
  {
    stopWatch.reset();

    syncLock.lock();

    if (GlobalVariable::STITCHING_MODE_NUM == 1)
    {
      if (heartBeat[0] == 0)
      {
        heartBeat[0] = 1;

        if (GlobalVariable::Left_FineTune_Trigger == true)
        {
          Eigen::Matrix4f final_transform_tmp;

          GlobalVariable::Left_FineTune_Trigger = false;
          LSA.setInitTransform(GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1], GlobalVariable::UI_PARA[2],
                               GlobalVariable::UI_PARA[3], GlobalVariable::UI_PARA[4], GlobalVariable::UI_PARA[5]);
          LSA.updateEstimation(cloudLeft, cloudTop);
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

          GlobalVariable::UI_PARA[0] = final_transform_tmp(0, 3);
          GlobalVariable::UI_PARA[1] = final_transform_tmp(1, 3);
          GlobalVariable::UI_PARA[2] = final_transform_tmp(2, 3);
          GlobalVariable::UI_PARA[3] = ea(0);
          GlobalVariable::UI_PARA[4] = ea(1);
          GlobalVariable::UI_PARA[5] = ea(2);
        }

        *cloudLeft = Transform_CUDA().compute<PointXYZI>(
            cloudLeft, GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1], GlobalVariable::UI_PARA[2],
            GlobalVariable::UI_PARA[3], GlobalVariable::UI_PARA[4], GlobalVariable::UI_PARA[5]);
      }
      else
      {
        heartBeat[0]++;
        if (heartBeat[0] > CHECKTIMES)
        {
          cloudLeft->clear();
        }
      }

      if (heartBeat[1] == 0)
      {
        heartBeat[1] = 1;

        if (GlobalVariable::Right_FineTune_Trigger == true)
        {
          Eigen::Matrix4f final_transform_tmp;

          GlobalVariable::Right_FineTune_Trigger = false;
          LSA.setInitTransform(GlobalVariable::UI_PARA[6], GlobalVariable::UI_PARA[7], GlobalVariable::UI_PARA[8],
                               GlobalVariable::UI_PARA[9], GlobalVariable::UI_PARA[10], GlobalVariable::UI_PARA[11]);
          LSA.updateEstimation(cloudRight, cloudTop);  // src, base
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
          cout << "to Euler angles:" << ea << endl;

          GlobalVariable::UI_PARA[6] = final_transform_tmp(0, 3);
          GlobalVariable::UI_PARA[7] = final_transform_tmp(1, 3);
          GlobalVariable::UI_PARA[8] = final_transform_tmp(2, 3);
          GlobalVariable::UI_PARA[9] = ea(0);
          GlobalVariable::UI_PARA[10] = ea(1);
          GlobalVariable::UI_PARA[11] = ea(2);
        }

        *cloudRight = Transform_CUDA().compute<PointXYZI>(
            cloudRight, GlobalVariable::UI_PARA[6], GlobalVariable::UI_PARA[7], GlobalVariable::UI_PARA[8],
            GlobalVariable::UI_PARA[9], GlobalVariable::UI_PARA[10], GlobalVariable::UI_PARA[11]);
      }
      else
      {
        heartBeat[1]++;
        if (heartBeat[1] > CHECKTIMES)
        {
          cloudRight->clear();
        }
      }

      if (heartBeat[2] == 0)
      {
        heartBeat[2] = 1;

        if (GlobalVariable::Front_FineTune_Trigger == true)
        {
          Eigen::Matrix4f final_transform_tmp;

          GlobalVariable::Front_FineTune_Trigger = false;
          LSA.setInitTransform(GlobalVariable::UI_PARA[12], GlobalVariable::UI_PARA[13], GlobalVariable::UI_PARA[14],
                               GlobalVariable::UI_PARA[15], GlobalVariable::UI_PARA[16], GlobalVariable::UI_PARA[17]);
          LSA.updateEstimation(cloudFront, cloudTop);  // src, base
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
          cout << "to Euler angles:" << ea << endl;

          GlobalVariable::UI_PARA[12] = final_transform_tmp(0, 3);
          GlobalVariable::UI_PARA[13] = final_transform_tmp(1, 3);
          GlobalVariable::UI_PARA[14] = final_transform_tmp(2, 3);
          GlobalVariable::UI_PARA[15] = ea(0);
          GlobalVariable::UI_PARA[16] = ea(1);
          GlobalVariable::UI_PARA[17] = ea(2);
        }

        *cloudFront = Transform_CUDA().compute<PointXYZI>(
            cloudFront, GlobalVariable::UI_PARA[12], GlobalVariable::UI_PARA[13], GlobalVariable::UI_PARA[14],
            GlobalVariable::UI_PARA[15], GlobalVariable::UI_PARA[16], GlobalVariable::UI_PARA[17]);
      }
      else
      {
        heartBeat[2]++;
        if (heartBeat[2] > CHECKTIMES)
        {
          cloudFront->clear();
        }
      }

      if (heartBeat[3] == 0)
      {
        heartBeat[3] = 1;
        *cloudTop = Transform_CUDA().compute<PointXYZI>(cloudTop, 0, 0, 0, 0, 0, 0);
      }
      else
      {
        heartBeat[3]++;
        if (heartBeat[3] > CHECKTIMES)
        {
          cloudTop->clear();
        }
      }
    }
    else  //=================================================================== Default Mode
    {
      if (heartBeat[0] == 0)
      {
        heartBeat[0] = 1;
        *cloudLeft = Transform_CUDA().compute<PointXYZI>(cloudLeft, LidarLeft_Fine_Param[0], LidarLeft_Fine_Param[1],
                                                         LidarLeft_Fine_Param[2], LidarLeft_Fine_Param[3],
                                                         LidarLeft_Fine_Param[4], LidarLeft_Fine_Param[5]);
      }
      else
      {
        heartBeat[0]++;
        if (heartBeat[0] > CHECKTIMES)
        {
          cloudLeft->clear();
        }
      }

      if (heartBeat[1] == 0)
      {
        heartBeat[1] = 1;
        *cloudRight = Transform_CUDA().compute<PointXYZI>(
            cloudRight, LidarRight_Fine_Param[0], LidarRight_Fine_Param[1], LidarRight_Fine_Param[2],
            LidarRight_Fine_Param[3], LidarRight_Fine_Param[4], LidarRight_Fine_Param[5]);
      }
      else
      {
        heartBeat[1]++;
        if (heartBeat[1] > CHECKTIMES)
        {
          cloudRight->clear();
        }
      }

      if (heartBeat[2] == 0)
      {
        heartBeat[2] = 1;
        *cloudFront = Transform_CUDA().compute<PointXYZI>(
            cloudFront, LidarFront_Fine_Param[0], LidarFront_Fine_Param[1], LidarFront_Fine_Param[2],
            LidarFront_Fine_Param[3], LidarFront_Fine_Param[4], LidarFront_Fine_Param[5]);
      }
      else
      {
        heartBeat[2]++;
        if (heartBeat[2] > CHECKTIMES)
        {
          cloudFront->clear();
        }
      }

      if (heartBeat[3] == 0)
      {
        heartBeat[3] = 1;
        *cloudTop = Transform_CUDA().compute<PointXYZI>(cloudTop, 0, 0, 0, 0, 0, 0);
      }
      else
      {
        heartBeat[3]++;
        if (heartBeat[3] > CHECKTIMES)
        {
          cloudTop->clear();
        }
      }
    }

    *cloudLidarAll = *cloudLeft;
    *cloudLidarAll += *cloudRight;
    *cloudLidarAll += *cloudFront;
    *cloudLidarAll += *cloudTop;

    pcl::uint64_t maxPCLtime = max(max(cloudLeft->header.stamp, cloudRight->header.stamp),
                                   max(cloudFront->header.stamp, cloudTop->header.stamp));
    RosModuleHINO::send_LidarAll(*cloudLeft, *cloudRight, *cloudFront, *cloudTop, *cloudLidarAll, maxPCLtime, "lidar");

    syncLock.unlock();

    if (stopWatch.getTimeSeconds() > 0.01)
    {
      cout << "[Grabber slow]:" << stopWatch.getTimeSeconds() << "s" << endl;
    }

    ros::spinOnce();
    loop_rate.sleep();
  }

  cout << "[" << ros::this_node::getName() << "] "
       << "----------------------------end" << endl;
  return 0;
}
