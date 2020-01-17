#include "all_header.h"
#include "GlobalVariable.h"
#include "UI/QtViewer.h"
#include "Transform_CUDA.h"
#include "LiDARStitchingAuto.h"
#include "CuboidFilter.h"

pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontLeft (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontRight (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarRearLeft (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarRearRight (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontTop (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidAll (new pcl::PointCloud<pcl::PointXYZI>);

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
vector<double> Zero_Param (6, 0.0);

LiDARStitchingAuto LSA;

mutex syncLock;

bool heartBeat[5] = { false, false, false, false, false };  //{ FrontLeft, FrontRight, RearLeft, RearRight, FrontTop }
int heartBeat_times[5] = { 0, 0, 0, 0, 0 };
int lidarAll_pubFlag = 4;

void
syncLock_callback ();
void
checkPubFlag (int lidarNum);
void
lidarAll_Pub (int lidarNum);

void
cloud_cb_LidarFrontLeft (const boost::shared_ptr<const sensor_msgs::PointCloud2> &input_cloud)
{
  heartBeat[0] = true;
  pcl::fromROSMsg (*input_cloud, *cloudPtr_LidarFrontLeft);
  syncLock_callback ();
}
void
cloud_cb_LidarFrontRight (const boost::shared_ptr<const sensor_msgs::PointCloud2> &input_cloud)
{
  heartBeat[1] = true;
  pcl::fromROSMsg (*input_cloud, *cloudPtr_LidarFrontRight);
  syncLock_callback ();
}
void
cloud_cb_LidarRearLeft (const boost::shared_ptr<const sensor_msgs::PointCloud2> &input_cloud)
{
  heartBeat[2] = true;
  pcl::fromROSMsg (*input_cloud, *cloudPtr_LidarRearLeft);
  syncLock_callback ();

}
void
cloud_cb_LidarRearRight (const boost::shared_ptr<const sensor_msgs::PointCloud2> &input_cloud)
{
  heartBeat[3] = true;
  pcl::fromROSMsg (*input_cloud, *cloudPtr_LidarRearRight);
  syncLock_callback ();
}
void
cloud_cb_LidarFrontTop (const boost::shared_ptr<const sensor_msgs::PointCloud2> &input_cloud)
{
#if 0
  static ros::Time oldtimestamp;
  ros::Duration intervaltime = input_cloud->header.stamp - oldtimestamp;
  if(oldtimestamp.sec != 0 && intervaltime.toSec() > 0.1 && intervaltime.sec < 10)
  {
    cout << "[lidars_grabber]: missing data " << intervaltime << "," << input_cloud->header.stamp <<endl;
  }
  oldtimestamp = input_cloud->header.stamp;
#endif

  heartBeat[4] = true;
  pcl::fromROSMsg (*input_cloud, *cloudPtr_LidarFrontTop);
  syncLock_callback ();
}

void
lidarAll_Pub (int lidarNum)
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
    case 2:
      cloudPtr_LidAll->header.seq = cloudPtr_LidarRearLeft->header.seq;
      cloudPtr_LidAll->header.frame_id = cloudPtr_LidarRearLeft->header.frame_id;
      break;
    case 3:
      cloudPtr_LidAll->header.seq = cloudPtr_LidarRearRight->header.seq;
      cloudPtr_LidAll->header.frame_id = cloudPtr_LidarRearRight->header.frame_id;
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
  {
    n += 1;
  };
  if (cloudPtr_LidarFrontRight->header.stamp != 0)
  {
    n += 1;
  };
  if (cloudPtr_LidarRearLeft->header.stamp != 0)
  {
    n += 1;
  };
  if (cloudPtr_LidarRearRight->header.stamp != 0)
  {
    n += 1;
  };
  if (cloudPtr_LidarFrontTop->header.stamp != 0)
  {
    n += 1;
  };
  if (n != 0)
  {
    avg_time = (cloudPtr_LidarFrontLeft->header.stamp + cloudPtr_LidarFrontRight->header.stamp + cloudPtr_LidarRearLeft->header.stamp
        + cloudPtr_LidarRearRight->header.stamp + cloudPtr_LidarFrontTop->header.stamp) / n;
  }
  else
  {
    avg_time = 0;
  }

  cloudPtr_LidAll->header.stamp = avg_time;
  pub_LidAll.publish (*cloudPtr_LidAll);

  cloudPtr_LidAll->clear ();

  // if wall_time - ros_time !> 30 minutes, (not rosbag)
  // clear sensor pc data memory if delay 3sec. 
  uint64_t now = ros::Time::now ().toNSec () / 1000ull;  //microsec
  if (! ( (now - avg_time) > 1000000 * 1800))
  {

    if ( (now - cloudPtr_LidarFrontLeft->header.stamp) > 1000000 * 3)
    {
      cloudPtr_LidarFrontLeft->clear ();
      cout << "Front-Left Clear" << endl;
    };
    if ( (now - cloudPtr_LidarFrontRight->header.stamp) > 1000000 * 3)
    {
      cloudPtr_LidarFrontRight->clear ();
      cout << "Front-Right Clear" << endl;
    };
    if ( (now - cloudPtr_LidarRearLeft->header.stamp) > 1000000 * 3)
    {
      cloudPtr_LidarRearLeft->clear ();
      cout << "Rear-Left Clear" << endl;
    };
    if ( (now - cloudPtr_LidarRearRight->header.stamp) > 1000000 * 3)
    {
      cloudPtr_LidarRearRight->clear ();
      cout << "Rear-Right Clear" << endl;
    };
    if ( (now - cloudPtr_LidarFrontTop->header.stamp) > 1000000 * 3)
    {
      cloudPtr_LidarFrontTop->clear ();
      cout << "Top Clear" << endl;
    };

  }

}

void
checkPubFlag (int lidarNum)
{
  if (lidarAll_pubFlag == lidarNum)
  {
    //cout << "[PubFlag]: " << lidarNum << endl;
    lidarAll_Pub (lidarNum);
    heartBeat_times[0] = 0;
    heartBeat_times[1] = 0;
    heartBeat_times[2] = 0;
    heartBeat_times[3] = 0;
    heartBeat_times[4] = 0;

  }
  else if (lidarAll_pubFlag != lidarNum && heartBeat_times[lidarNum] > 3)
  {
    lidarAll_pubFlag = lidarNum;
    //cout << "[PubFlag]: " << lidarNum << endl;
    lidarAll_Pub (lidarNum);
    heartBeat_times[0] = 0;
    heartBeat_times[1] = 0;
    heartBeat_times[2] = 0;
    heartBeat_times[3] = 0;
    heartBeat_times[4] = 0;
  }
  else
  {
    heartBeat_times[lidarNum] += 1;
  }
}

void
syncLock_callback ()
{
  syncLock.lock ();

  if (GlobalVariable::STITCHING_MODE_NUM == 1)
  {

    if (heartBeat[0] == true)
    {
      heartBeat[0] = false;

      if (GlobalVariable::FrontLeft_FineTune_Trigger == true)
      {
        Eigen::Matrix4f final_transform_tmp;

        GlobalVariable::FrontLeft_FineTune_Trigger = false;
        LSA.setInitTransform (GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1], GlobalVariable::UI_PARA[2], GlobalVariable::UI_PARA[3],
                              GlobalVariable::UI_PARA[4], GlobalVariable::UI_PARA[5]);
        LSA.updateEstimation (cloudPtr_LidarFrontLeft, cloudPtr_LidarFrontTop);  //src, base
        LSA.getFinalTransform (final_transform_tmp);

        cout << final_transform_tmp << endl;

        Eigen::Matrix3f m;
        m (0, 0) = final_transform_tmp (0, 0);
        m (0, 1) = final_transform_tmp (0, 1);
        m (0, 2) = final_transform_tmp (0, 2);
        m (1, 0) = final_transform_tmp (1, 0);
        m (1, 1) = final_transform_tmp (1, 1);
        m (1, 2) = final_transform_tmp (1, 2);
        m (2, 0) = final_transform_tmp (2, 0);
        m (2, 1) = final_transform_tmp (2, 1);
        m (2, 2) = final_transform_tmp (2, 2);

        Eigen::Vector3f ea = m.eulerAngles (0, 1, 2);
        cout << "to Euler angles:" << endl;
        cout << ea << endl;

        //write to GlobalVariable::UI_PARA[0~6]
        GlobalVariable::UI_PARA[0] = final_transform_tmp (0, 3);
        GlobalVariable::UI_PARA[1] = final_transform_tmp (1, 3);
        GlobalVariable::UI_PARA[2] = final_transform_tmp (2, 3);
        GlobalVariable::UI_PARA[3] = ea (0);
        GlobalVariable::UI_PARA[4] = ea (1);
        GlobalVariable::UI_PARA[5] = ea (2);

      }

      *cloudPtr_LidarFrontLeft = Transform_CUDA ().compute<PointXYZI> (cloudPtr_LidarFrontLeft, GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1],
                                                                       GlobalVariable::UI_PARA[2], GlobalVariable::UI_PARA[3], GlobalVariable::UI_PARA[4],
                                                                       GlobalVariable::UI_PARA[5]);
      cloudPtr_LidarFrontLeft->header.frame_id = "lidar";
      pub_LidarFrontLeft.publish (*cloudPtr_LidarFrontLeft);
      checkPubFlag (0);
    }

    if (heartBeat[1] == true)
    {
      heartBeat[1] = false;

      if (GlobalVariable::FrontRight_FineTune_Trigger == true)
      {
        Eigen::Matrix4f final_transform_tmp;

        GlobalVariable::FrontRight_FineTune_Trigger = false;
        LSA.setInitTransform (GlobalVariable::UI_PARA[6], GlobalVariable::UI_PARA[7], GlobalVariable::UI_PARA[8], GlobalVariable::UI_PARA[9],
                              GlobalVariable::UI_PARA[10], GlobalVariable::UI_PARA[11]);
        LSA.updateEstimation (cloudPtr_LidarFrontRight, cloudPtr_LidarFrontTop);  //src, base
        LSA.getFinalTransform (final_transform_tmp);

        cout << final_transform_tmp << endl;

        Eigen::Matrix3f m;
        m (0, 0) = final_transform_tmp (0, 0);
        m (0, 1) = final_transform_tmp (0, 1);
        m (0, 2) = final_transform_tmp (0, 2);
        m (1, 0) = final_transform_tmp (1, 0);
        m (1, 1) = final_transform_tmp (1, 1);
        m (1, 2) = final_transform_tmp (1, 2);
        m (2, 0) = final_transform_tmp (2, 0);
        m (2, 1) = final_transform_tmp (2, 1);
        m (2, 2) = final_transform_tmp (2, 2);

        Eigen::Vector3f ea = m.eulerAngles (0, 1, 2);
        cout << "to Euler angles:" << endl;
        cout << ea << endl;

        //write to GlobalVariable::UI_PARA[0~6]
        GlobalVariable::UI_PARA[6] = final_transform_tmp (0, 3);
        GlobalVariable::UI_PARA[7] = final_transform_tmp (1, 3);
        GlobalVariable::UI_PARA[8] = final_transform_tmp (2, 3);
        GlobalVariable::UI_PARA[9] = ea (0);
        GlobalVariable::UI_PARA[10] = ea (1);
        GlobalVariable::UI_PARA[11] = ea (2);

      }

      *cloudPtr_LidarFrontRight = Transform_CUDA ().compute<PointXYZI> (cloudPtr_LidarFrontRight, GlobalVariable::UI_PARA[6], GlobalVariable::UI_PARA[7],
                                                                        GlobalVariable::UI_PARA[8], GlobalVariable::UI_PARA[9], GlobalVariable::UI_PARA[10],
                                                                        GlobalVariable::UI_PARA[11]);

      cloudPtr_LidarFrontRight->header.frame_id = "lidar";
      pub_LidarFrontRight.publish (*cloudPtr_LidarFrontRight);
      checkPubFlag (1);

    }

    if (heartBeat[2] == true)
    {
      heartBeat[2] = false;

      if (GlobalVariable::RearLeft_FineTune_Trigger == true)
      {
        Eigen::Matrix4f final_transform_tmp;

        GlobalVariable::RearLeft_FineTune_Trigger = false;
        LSA.setInitTransform (GlobalVariable::UI_PARA[12], GlobalVariable::UI_PARA[13], GlobalVariable::UI_PARA[14], GlobalVariable::UI_PARA[15],
                              GlobalVariable::UI_PARA[16], GlobalVariable::UI_PARA[17]);
        LSA.updateEstimation (cloudPtr_LidarRearLeft, cloudPtr_LidarFrontTop);  //src, base
        LSA.getFinalTransform (final_transform_tmp);

        cout << final_transform_tmp << endl;

        Eigen::Matrix3f m;
        m (0, 0) = final_transform_tmp (0, 0);
        m (0, 1) = final_transform_tmp (0, 1);
        m (0, 2) = final_transform_tmp (0, 2);
        m (1, 0) = final_transform_tmp (1, 0);
        m (1, 1) = final_transform_tmp (1, 1);
        m (1, 2) = final_transform_tmp (1, 2);
        m (2, 0) = final_transform_tmp (2, 0);
        m (2, 1) = final_transform_tmp (2, 1);
        m (2, 2) = final_transform_tmp (2, 2);

        Eigen::Vector3f ea = m.eulerAngles (0, 1, 2);
        cout << "to Euler angles:" << endl;
        cout << ea << endl;

        //write to GlobalVariable::UI_PARA[0~6]
        GlobalVariable::UI_PARA[12] = final_transform_tmp (0, 3);
        GlobalVariable::UI_PARA[13] = final_transform_tmp (1, 3);
        GlobalVariable::UI_PARA[14] = final_transform_tmp (2, 3);
        GlobalVariable::UI_PARA[15] = ea (0);
        GlobalVariable::UI_PARA[16] = ea (1);
        GlobalVariable::UI_PARA[17] = ea (2);

      }

      *cloudPtr_LidarRearLeft = Transform_CUDA ().compute<PointXYZI> (cloudPtr_LidarRearLeft, GlobalVariable::UI_PARA[12], GlobalVariable::UI_PARA[13],
                                                                      GlobalVariable::UI_PARA[14], GlobalVariable::UI_PARA[15], GlobalVariable::UI_PARA[16],
                                                                      GlobalVariable::UI_PARA[17]);
      cloudPtr_LidarRearLeft->header.frame_id = "lidar";
      pub_LidarRearLeft.publish (*cloudPtr_LidarRearLeft);
      checkPubFlag (2);

    }

    if (heartBeat[3] == true)
    {
      heartBeat[3] = false;

      if (GlobalVariable::RearRight_FineTune_Trigger == true)
      {
        Eigen::Matrix4f final_transform_tmp;

        GlobalVariable::RearRight_FineTune_Trigger = false;
        LSA.setInitTransform (GlobalVariable::UI_PARA[18], GlobalVariable::UI_PARA[19], GlobalVariable::UI_PARA[20], GlobalVariable::UI_PARA[21],
                              GlobalVariable::UI_PARA[22], GlobalVariable::UI_PARA[23]);
        LSA.updateEstimation (cloudPtr_LidarRearRight, cloudPtr_LidarFrontTop);  //src, base
        LSA.getFinalTransform (final_transform_tmp);

        cout << final_transform_tmp << endl;

        Eigen::Matrix3f m;
        m (0, 0) = final_transform_tmp (0, 0);
        m (0, 1) = final_transform_tmp (0, 1);
        m (0, 2) = final_transform_tmp (0, 2);
        m (1, 0) = final_transform_tmp (1, 0);
        m (1, 1) = final_transform_tmp (1, 1);
        m (1, 2) = final_transform_tmp (1, 2);
        m (2, 0) = final_transform_tmp (2, 0);
        m (2, 1) = final_transform_tmp (2, 1);
        m (2, 2) = final_transform_tmp (2, 2);

        Eigen::Vector3f ea = m.eulerAngles (0, 1, 2);
        cout << "to Euler angles:" << endl;
        cout << ea << endl;

        //write to GlobalVariable::UI_PARA[18-23]
        GlobalVariable::UI_PARA[18] = final_transform_tmp (0, 3);
        GlobalVariable::UI_PARA[19] = final_transform_tmp (1, 3);
        GlobalVariable::UI_PARA[20] = final_transform_tmp (2, 3);
        GlobalVariable::UI_PARA[21] = ea (0);
        GlobalVariable::UI_PARA[22] = ea (1);
        GlobalVariable::UI_PARA[23] = ea (2);

      }

      *cloudPtr_LidarRearRight = Transform_CUDA ().compute<PointXYZI> (cloudPtr_LidarRearRight, GlobalVariable::UI_PARA[18], GlobalVariable::UI_PARA[19],
                                                                       GlobalVariable::UI_PARA[20], GlobalVariable::UI_PARA[21], GlobalVariable::UI_PARA[22],
                                                                       GlobalVariable::UI_PARA[23]);
      cloudPtr_LidarRearRight->header.frame_id = "lidar";
      pub_LidarRearRight.publish (*cloudPtr_LidarRearRight);
      checkPubFlag (3);

    }

    // LidarFrontTop does not need to compute
    if (heartBeat[4] == true)
    {
      heartBeat[4] = false;

      *cloudPtr_LidarFrontTop = Transform_CUDA ().compute<PointXYZI> (cloudPtr_LidarFrontTop, 0, 0, 0, 0, 0.23, -1.61);

      cloudPtr_LidarFrontTop->header.frame_id = "lidar";

      pub_LidarFrontTop.publish (*cloudPtr_LidarFrontTop);
      checkPubFlag (4);

    }

  }
  //========================== ELSE AREA ==========================//
  else
  {
    if (heartBeat[0] == true)
    {
      heartBeat[0] = false;
      *cloudPtr_LidarFrontLeft = Transform_CUDA ().compute<PointXYZI> (cloudPtr_LidarFrontLeft, LidarFrontLeft_Fine_Param[0], LidarFrontLeft_Fine_Param[1],
                                                                       LidarFrontLeft_Fine_Param[2], LidarFrontLeft_Fine_Param[3], LidarFrontLeft_Fine_Param[4],
                                                                       LidarFrontLeft_Fine_Param[5]);
      cloudPtr_LidarFrontLeft->header.frame_id = "lidar";
      pub_LidarFrontLeft.publish (*cloudPtr_LidarFrontLeft);
      checkPubFlag (0);

    }
    if (heartBeat[1] == true)
    {
      heartBeat[1] = false;
      *cloudPtr_LidarFrontRight = Transform_CUDA ().compute<PointXYZI> (cloudPtr_LidarFrontRight, LidarFrontRight_Fine_Param[0], LidarFrontRight_Fine_Param[1],
                                                                        LidarFrontRight_Fine_Param[2], LidarFrontRight_Fine_Param[3],
                                                                        LidarFrontRight_Fine_Param[4], LidarFrontRight_Fine_Param[5]);
      cloudPtr_LidarFrontRight->header.frame_id = "lidar";
      pub_LidarFrontRight.publish (*cloudPtr_LidarFrontRight);
      checkPubFlag (1);

    }
    if (heartBeat[2] == true)
    {
      heartBeat[2] = false;
      *cloudPtr_LidarRearLeft = Transform_CUDA ().compute<PointXYZI> (cloudPtr_LidarRearLeft, LidarRearLeft_Fine_Param[0], LidarRearLeft_Fine_Param[1],
                                                                      LidarRearLeft_Fine_Param[2], LidarRearLeft_Fine_Param[3], LidarRearLeft_Fine_Param[4],
                                                                      LidarRearLeft_Fine_Param[5]);
      cloudPtr_LidarRearLeft->header.frame_id = "lidar";
      pub_LidarRearLeft.publish (*cloudPtr_LidarRearLeft);

      checkPubFlag (2);
    }
    if (heartBeat[3] == true)
    {
      heartBeat[3] = false;
      *cloudPtr_LidarRearRight = Transform_CUDA ().compute<PointXYZI> (cloudPtr_LidarRearRight, LidarRearRight_Fine_Param[0], LidarRearRight_Fine_Param[1],
                                                                       LidarRearRight_Fine_Param[2], LidarRearRight_Fine_Param[3], LidarRearRight_Fine_Param[4],
                                                                       LidarRearRight_Fine_Param[5]);
      *cloudPtr_LidarRearRight = CuboidFilter ().hollow_removal<PointXYZI> (cloudPtr_LidarRearRight, 0, 50, -0.8, 20, -5, 5);
      cloudPtr_LidarRearRight->header.frame_id = "lidar";
      pub_LidarRearRight.publish (*cloudPtr_LidarRearRight);
      checkPubFlag (3);

    }

    if (heartBeat[4] == true)
    {
      heartBeat[4] = false;
      *cloudPtr_LidarFrontTop = Transform_CUDA ().compute<PointXYZI> (cloudPtr_LidarFrontTop, 0, 0, 0, 0, 0.23, -1.61);
      *cloudPtr_LidarFrontTop = CuboidFilter ().hollow_removal<PointXYZI> (cloudPtr_LidarFrontTop, -20, -2, -2, 2, -5, 0);
      cloudPtr_LidarFrontTop->header.frame_id = "lidar";
      pub_LidarFrontTop.publish (*cloudPtr_LidarFrontTop);
      checkPubFlag (4);
    }
  }

  syncLock.unlock ();
}

void
UI (int argc,
    char ** argv)
{
  if (pcl::console::find_switch (argc, argv, "-D"))
  {
    QApplication a (argc, argv);
    QtViewer w;
    w.show ();
    a.exec ();
  }
}

int
main (int argc,
      char **argv)
{
  cout << "=============== Grabber Start ===============" << endl;

  ros::init (argc, argv, "lidars_grabber");
  ros::NodeHandle n;

  if (pcl::console::find_switch (argc, argv, "-D"))
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

  if (!ros::param::has ("/LidarFrontRight_Fine_Param"))
  {
    n.setParam ("LidarFrontLeft_Fine_Param", Zero_Param);
    n.setParam ("LidarFrontRight_Fine_Param", Zero_Param);
    n.setParam ("LidarRearLeft_Fine_Param", Zero_Param);
    n.setParam ("LidarRearRight_Fine_Param", Zero_Param);

    cout << "NO STITCHING PARAMETER INPUT!" << endl;
    cout << "Now is using [0,0,0,0,0,0] as stitching parameter!" << endl;
  }
  else
  {
    n.param ("/LidarFrontLeft_Fine_Param", LidarFrontLeft_Fine_Param, vector<double> ());
    n.param ("/LidarFrontRight_Fine_Param", LidarFrontRight_Fine_Param, vector<double> ());
    n.param ("/LidarRearLeft_Fine_Param", LidarRearLeft_Fine_Param, vector<double> ());
    n.param ("/LidarRearRight_Fine_Param", LidarRearRight_Fine_Param, vector<double> ());

    cout << "STITCHING PARAMETER FIND!" << endl;

    cout << "LidarFrontLeft is using  " << "[" << LidarFrontLeft_Fine_Param[0] << ", " << LidarFrontLeft_Fine_Param[1] << ", " << LidarFrontLeft_Fine_Param[2]
        << ", " << LidarFrontLeft_Fine_Param[3] << ", " << LidarFrontLeft_Fine_Param[4] << ", " << LidarFrontLeft_Fine_Param[5] << "]" << endl;

    cout << "LidarFrontRight is using " << "[" << LidarFrontRight_Fine_Param[0] << ", " << LidarFrontRight_Fine_Param[1] << ", "
        << LidarFrontRight_Fine_Param[2] << ", " << LidarFrontRight_Fine_Param[3] << ", " << LidarFrontRight_Fine_Param[4] << ", "
        << LidarFrontRight_Fine_Param[5] << "]" << endl;

    cout << "LidarRearLeft is using " << "[" << LidarRearLeft_Fine_Param[0] << ", " << LidarRearLeft_Fine_Param[1] << ", " << LidarRearLeft_Fine_Param[2]
        << ", " << LidarRearLeft_Fine_Param[3] << ", " << LidarRearLeft_Fine_Param[4] << ", " << LidarRearLeft_Fine_Param[5] << "]" << endl;

    cout << "LidarRearRight is using " << "[" << LidarRearRight_Fine_Param[0] << ", " << LidarRearRight_Fine_Param[1] << ", " << LidarRearRight_Fine_Param[2]
        << ", " << LidarRearRight_Fine_Param[3] << ", " << LidarRearRight_Fine_Param[4] << ", " << LidarRearRight_Fine_Param[5] << "]" << endl;

  }

  ros::Subscriber sub_LidarFrontLeft = n.subscribe<sensor_msgs::PointCloud2> ("/LidarFrontLeft/Raw", 1, cloud_cb_LidarFrontLeft);
  ros::Subscriber sub_LidarFrontRight = n.subscribe<sensor_msgs::PointCloud2> ("/LidarFrontRight/Raw", 1, cloud_cb_LidarFrontRight);
  ros::Subscriber sub_LidarRearLeft = n.subscribe<sensor_msgs::PointCloud2> ("/LidarRearLeft/Raw", 1, cloud_cb_LidarRearLeft);
  ros::Subscriber sub_LidarRearRight = n.subscribe<sensor_msgs::PointCloud2> ("/LidarRearRight/Raw", 1, cloud_cb_LidarRearRight);
  ros::Subscriber sub_LidarFrontTop = n.subscribe<sensor_msgs::PointCloud2> ("/LidarFrontTop/Raw", 1, cloud_cb_LidarFrontTop);

  pub_LidarFrontLeft = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarFrontLeft", 1);
  pub_LidarFrontRight = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarFrontRight", 1);
  pub_LidarRearLeft = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarRearLeft", 1);
  pub_LidarRearRight = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarRearRight", 1);
  pub_LidarFrontTop = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarFrontTop", 1);
  pub_LidAll = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarAll", 1);

  thread TheadDetection (UI, argc, argv);

  ros::Rate loop_rate (80);  //80Hz
  while (ros::ok ())
  {
    ros::spinOnce ();
    loop_rate.sleep ();
  }

  cout << "=============== Grabber Stop ===============" << endl;
  return 0;
}

