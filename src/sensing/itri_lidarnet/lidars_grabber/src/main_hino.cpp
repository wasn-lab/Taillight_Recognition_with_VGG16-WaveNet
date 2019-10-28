#include "all_header.h"
#include "GlobalVariable.h"

#include "Transform_CUDA.h"


pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidLeft (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidRight (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidFront (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidTop (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidAll (new pcl::PointCloud<pcl::PointXYZI>);

ros::Publisher pub_LidLeft;
ros::Publisher pub_LidRight;
ros::Publisher pub_LidFront;
ros::Publisher pub_LidTop;
ros::Publisher pub_LidAll;

vector<double> LidarLeft_Fine_Param;
vector<double> LidarRight_Fine_Param;
vector<double> LidarFront_Fine_Param;
vector<double> Zero_Param (6, 0.0);



mutex syncLock;

//left, right, front, top
bool heartBeat[5] = { false, false, false, false };
int heartBeat_times[4] = { 0, 0, 0, 0 };
int lidarAll_pubFlag = 3;

void
lidarAll_Pub (int lidarNum)
{
  switch (lidarNum)
  {
    case 0:
      cloudPtr_LidAll->header.seq = cloudPtr_LidLeft->header.seq;
      cloudPtr_LidAll->header.frame_id = cloudPtr_LidLeft->header.frame_id;
      break;
    case 1:
      cloudPtr_LidAll->header.seq = cloudPtr_LidRight->header.seq;
      cloudPtr_LidAll->header.frame_id = cloudPtr_LidRight->header.frame_id;
      break;
    case 2:
      cloudPtr_LidAll->header.seq = cloudPtr_LidFront->header.seq;
      cloudPtr_LidAll->header.frame_id = cloudPtr_LidFront->header.frame_id;
      break;
    case 3:
      cloudPtr_LidAll->header.seq = cloudPtr_LidTop->header.seq;
      cloudPtr_LidAll->header.frame_id = cloudPtr_LidTop->header.frame_id;
      break;
    default:
      cloudPtr_LidAll->header.seq = cloudPtr_LidTop->header.seq;
      cloudPtr_LidAll->header.frame_id = cloudPtr_LidTop->header.frame_id;
      break;
  }

  *cloudPtr_LidAll = *cloudPtr_LidLeft;
  *cloudPtr_LidAll += *cloudPtr_LidRight;
  *cloudPtr_LidAll += *cloudPtr_LidFront;
  *cloudPtr_LidAll += *cloudPtr_LidTop;

  uint64_t avg_time;
  int n = 0;
  if (cloudPtr_LidLeft->header.stamp != 0)
  {
    n += 1;
  };
  if (cloudPtr_LidRight->header.stamp != 0)
  {
    n += 1;
  };
  if (cloudPtr_LidFront->header.stamp != 0)
  {
    n += 1;
  };
  if (cloudPtr_LidTop->header.stamp != 0)
  {
    n += 1;
  };
  if (n != 0)
  {
    avg_time = (cloudPtr_LidLeft->header.stamp + cloudPtr_LidRight->header.stamp + cloudPtr_LidFront->header.stamp + cloudPtr_LidTop->header.stamp) / n;
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

    if ( (now - cloudPtr_LidLeft->header.stamp) > 1000000 * 3)
    {
      cloudPtr_LidLeft->clear ();
      cout << "Left Clear" << endl;
    };
    if ( (now - cloudPtr_LidRight->header.stamp) > 1000000 * 3)
    {
      cloudPtr_LidRight->clear ();
      cout << "Right Clear" << endl;
    };
    if ( (now - cloudPtr_LidFront->header.stamp) > 1000000 * 3)
    {
      cloudPtr_LidFront->clear ();
      cout << "Front Clear" << endl;
    };
    if ( (now - cloudPtr_LidTop->header.stamp) > 1000000 * 3)
    {
      cloudPtr_LidTop->clear ();
      cout << "Top Clear" << endl;
    };
  }
}

void
checkPubFlag (int lidarNum)
{
  if (lidarAll_pubFlag == lidarNum)
  {
    lidarAll_Pub (lidarNum);
    heartBeat_times[0] = 0;
    heartBeat_times[1] = 0;
    heartBeat_times[2] = 0;
    heartBeat_times[3] = 0;
  }
  else if (lidarAll_pubFlag != lidarNum && heartBeat_times[lidarNum] > 3)
  {
    lidarAll_pubFlag = lidarNum;
    lidarAll_Pub (lidarNum);
    heartBeat_times[0] = 0;
    heartBeat_times[1] = 0;
    heartBeat_times[2] = 0;
    heartBeat_times[3] = 0;
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



  {
    StopWatch stopWatch;

    if (heartBeat[0] == true)
    {
      heartBeat[0] = false;
      *cloudPtr_LidLeft = Transform_CUDA ().compute (cloudPtr_LidLeft, LidarLeft_Fine_Param[0], LidarLeft_Fine_Param[1], LidarLeft_Fine_Param[2],
                                                     LidarLeft_Fine_Param[3], LidarLeft_Fine_Param[4], LidarLeft_Fine_Param[5]);
      cloudPtr_LidLeft->header.frame_id = "lidar";
      pub_LidLeft.publish (*cloudPtr_LidLeft);
      //ROS_INFO("Lidar0 published");
      checkPubFlag (0);
    }
    if (heartBeat[1] == true)
    {
      heartBeat[1] = false;
      *cloudPtr_LidRight = Transform_CUDA ().compute (cloudPtr_LidRight, LidarRight_Fine_Param[0], LidarRight_Fine_Param[1], LidarRight_Fine_Param[2],
                                                      LidarRight_Fine_Param[3], LidarRight_Fine_Param[4], LidarRight_Fine_Param[5]);
      cloudPtr_LidRight->header.frame_id = "lidar";
      pub_LidRight.publish (*cloudPtr_LidRight);
      //ROS_INFO("Lidar1 published");
      checkPubFlag (1);
    }
    if (heartBeat[2] == true)
    {
      heartBeat[2] = false;
      *cloudPtr_LidFront = Transform_CUDA ().compute (cloudPtr_LidFront, LidarFront_Fine_Param[0], LidarFront_Fine_Param[1], LidarFront_Fine_Param[2],
                                                      LidarFront_Fine_Param[3], LidarFront_Fine_Param[4], LidarFront_Fine_Param[5]);

      cloudPtr_LidFront->header.frame_id = "lidar";
      pub_LidFront.publish (*cloudPtr_LidFront);
      //ROS_INFO("Lidar2 published");
      checkPubFlag (2);
    }
    if (heartBeat[3] == true)
    {
      heartBeat[3] = false;
      *cloudPtr_LidTop = Transform_CUDA ().compute (cloudPtr_LidTop, 0, 0, 0, 0, 0, 0);
      cloudPtr_LidTop->header.frame_id = "lidar";
      pub_LidTop.publish (*cloudPtr_LidTop);
      //ROS_INFO("Lidar3 published");
      checkPubFlag (3);
    }

    if (stopWatch.getTimeSeconds () > 0.05)
    {
      cout << "[Grabber slow]:" << stopWatch.getTimeSeconds () << "s" << endl;
    }
  }

  syncLock.unlock ();
}

void
cloud_cb_LidLeft (const boost::shared_ptr<const sensor_msgs::PointCloud2> &input_cloud)
{
  heartBeat[0] = true;
  pcl::fromROSMsg (*input_cloud, *cloudPtr_LidLeft);
  syncLock_callback ();
}

void
cloud_cb_LidRight (const boost::shared_ptr<const sensor_msgs::PointCloud2> &input_cloud)
{
  heartBeat[1] = true;
  pcl::fromROSMsg (*input_cloud, *cloudPtr_LidRight);
  syncLock_callback ();
}

void
cloud_cb_LidFront (const boost::shared_ptr<const sensor_msgs::PointCloud2> &input_cloud)
{
  heartBeat[2] = true;
  pcl::fromROSMsg (*input_cloud, *cloudPtr_LidFront);
  syncLock_callback ();
}

void
cloud_cb_LidTop (const boost::shared_ptr<const sensor_msgs::PointCloud2> &input_cloud)
{
  heartBeat[3] = true;
  pcl::fromROSMsg (*input_cloud, *cloudPtr_LidTop);
  syncLock_callback ();
}



int
main (int argc,
      char **argv)
{
  cout << "=============== Grabber Start ===============" << endl;

  ros::init (argc, argv, "lidars_grabber");
  ros::NodeHandle n;

  ros::Subscriber sub_LidLeft = n.subscribe<sensor_msgs::PointCloud2> ("/LidarLeft/Raw", 1, cloud_cb_LidLeft);
  ros::Subscriber sub_LidRight = n.subscribe<sensor_msgs::PointCloud2> ("/LidarRight/Raw", 1, cloud_cb_LidRight);
  ros::Subscriber sub_LidFront = n.subscribe<sensor_msgs::PointCloud2> ("/LidarFront/Raw", 1, cloud_cb_LidFront);
  ros::Subscriber sub_LidTop = n.subscribe<sensor_msgs::PointCloud2> ("/LidarTop/Raw", 1, cloud_cb_LidTop);

  pub_LidLeft = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarLeft", 1);
  pub_LidRight = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarRight", 1);
  pub_LidFront = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarFront", 1);
  pub_LidTop = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarTop", 1);
  pub_LidAll = n.advertise<pcl::PointCloud<pcl::PointXYZI> > ("/LidarAll", 1);

  if (ros::param::has ("/LidarLeft_Fine_Param"))
  {
    n.param ("/LidarLeft_Fine_Param", LidarLeft_Fine_Param, vector<double> ());
    n.param ("/LidarRight_Fine_Param", LidarRight_Fine_Param, vector<double> ());
    n.param ("/LidarFront_Fine_Param", LidarFront_Fine_Param, vector<double> ());

    cout << "STITCHING PARAMETER FIND!" << endl;

    cout << "LidarLeft is using  " << "[" << LidarLeft_Fine_Param[0] << ", " << LidarLeft_Fine_Param[1] << ", " << LidarLeft_Fine_Param[2] << ", "
        << LidarLeft_Fine_Param[3] << ", " << LidarLeft_Fine_Param[4] << ", " << LidarLeft_Fine_Param[5] << "]" << endl;

    cout << "LidarRight is using " << "[" << LidarRight_Fine_Param[0] << ", " << LidarRight_Fine_Param[1] << ", " << LidarRight_Fine_Param[2] << ", "
        << LidarRight_Fine_Param[3] << ", " << LidarRight_Fine_Param[4] << ", " << LidarRight_Fine_Param[5] << "]" << endl;

    cout << "LidarFront is using " << "[" << LidarFront_Fine_Param[0] << ", " << LidarFront_Fine_Param[1] << ", " << LidarFront_Fine_Param[2] << ", "
        << LidarFront_Fine_Param[3] << ", " << LidarFront_Fine_Param[4] << ", " << LidarFront_Fine_Param[5] << "]" << endl;

  }
  else
  {
    n.setParam ("LidarLeft_Fine_Param", Zero_Param);
    n.setParam ("LidarRight_Fine_Param", Zero_Param);
    n.setParam ("LidarFront_Fine_Param", Zero_Param);

    cout << "NO STITCHING PARAMETER INPUT! Using [0,0,0,0,0,0] as stitching parameter" << endl;
  }

  if (pcl::console::find_switch (argc, argv, "-D"))
  {
    GlobalVariable::STITCHING_MODE_NUM = 1;
  }



  ros::Rate loop_rate (40);
  while (ros::ok ())
  {
    ros::spinOnce ();
    loop_rate.sleep ();
  }

  cout << "=============== Grabber Stop ===============" << endl;
  return 0;
}

