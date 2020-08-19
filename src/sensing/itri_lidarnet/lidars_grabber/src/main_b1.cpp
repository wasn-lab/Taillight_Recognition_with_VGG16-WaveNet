#include "all_header.h"
#include "GlobalVariable.h"
#include "UI/QtViewer.h"
#include "Transform_CUDA.h"
#include "LiDARStitchingAuto.h"
#include "CuboidFilter.h"
#include "NoiseFilter.h"
#include "extract_Indices.h"
#include "pointcloud_format_conversion.h"
#include "msgs/CompressedPointCloud.h"

//---------------------------- pointcloud
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontLeft(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontRight(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarFrontTop(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidarAll(new pcl::PointCloud<pcl::PointXYZI>);

//---------------------------- Publisher
// no-filter
ros::Publisher pub_LidarFrontLeft;
ros::Publisher pub_LidarFrontRight;
ros::Publisher pub_LidarFrontTop;
ros::Publisher pub_LidarAll;

ros::Publisher pub_LidarFrontLeft_Compress;
ros::Publisher pub_LidarFrontRight_Compress;
ros::Publisher pub_LidarFrontTop_Compress;

//----------------------------- Stitching
vector<double> LidarFrontLeft_Fine_Param;
vector<double> LidarFrontRight_Fine_Param;
vector<double> Zero_Param(6, 0.0);

LiDARStitchingAuto LSA;

//--------------------------- Global Variables
mutex L_Lock;
mutex R_Lock;
mutex T_Lock;

StopWatch stopWatch;
StopWatch stopWatch_L;
StopWatch stopWatch_R;
StopWatch stopWatch_T;

bool debug_output = false;
bool use_filter = false;
bool use_compress = false;
bool use_roi = false;

bool heartBeat[5] = { false, false, false, false, false };  //{ FrontLeft, FrontRight, RearLeft, RearRight, FrontTop }

void lidarAll_Pub(int lidarNum);

//------------------------------ Callback
void cloud_cb_LidarFrontLeft(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  L_Lock.lock();
  if (input_cloud->width * input_cloud->height > 100)
  {
    stopWatch_L.reset();
    heartBeat[0] = true;
    // check data from hardware
    if (debug_output && (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) < 3600)
    {
      uint64_t diff_time = (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) * 1000;
      cout << "[Left->Gbr]: " << diff_time << "ms" << endl;
    }
   
    //-------------------------- sensor_msg to pcl XYZIR
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_tmp(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZIR>::Ptr input_cloud_tmp_ring(new pcl::PointCloud<pcl::PointXYZIR>);
    *input_cloud_tmp_ring = SensorMsgs_to_XYZIR(*input_cloud, "velodyne");

    if (use_compress)
    {
      bool showStatistics = false;
      pcl::io::compression_Profiles_e compressionProfile = pcl::io::HIGH_RES_ONLINE_COMPRESSION_WITH_COLOR;
      pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA>* PointCloudEncoder;
      PointCloudEncoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA> (compressionProfile, showStatistics);

      // compressed stringstream
      msgs::CompressedPointCloud compressed_pointcloud;
      std::stringstream compressedData;

      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_xyzrgba(new pcl::PointCloud<pcl::PointXYZRGBA>);
      *cloud_xyzrgba = XYZIR_to_XYZRBGA(input_cloud_tmp_ring);


      PointCloudEncoder->encodePointCloud(cloud_xyzrgba, compressedData);
      compressed_pointcloud.data =  compressedData.str();

      compressed_pointcloud.header = input_cloud->header;

      pub_LidarFrontLeft_Compress.publish(compressed_pointcloud);
      
      delete (PointCloudEncoder);

      compressedData.clear();
      compressedData.str("");

    }

    //-------------------------- ring filter
    if (use_filter)
    {
      pcl::PointCloud<pcl::PointXYZIR>::Ptr output_cloud_tmp_ring(new pcl::PointCloud<pcl::PointXYZIR>);
      *output_cloud_tmp_ring = NoiseFilter().runRingOutlierRemoval(input_cloud_tmp_ring, 32, 0.3);
      pcl::copyPointCloud(*output_cloud_tmp_ring, *input_cloud_tmp);
    }
    else
    {
      pcl::copyPointCloud(*input_cloud_tmp_ring, *input_cloud_tmp);
    }

    // -------------------------- check stitching
    if (GlobalVariable::STITCHING_MODE_NUM == 1)
    {
      *cloudPtr_LidarFrontLeft = Transform_CUDA().compute<PointXYZI>(
          input_cloud_tmp, GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1], GlobalVariable::UI_PARA[2],
          GlobalVariable::UI_PARA[3], GlobalVariable::UI_PARA[4], GlobalVariable::UI_PARA[5]);

      cloudPtr_LidarFrontLeft->header.frame_id = "lidar";
      pub_LidarFrontLeft.publish(*cloudPtr_LidarFrontLeft);
    }
    else
    {
      heartBeat[0] = false;
      
      // Transfrom
      *input_cloud_tmp = Transform_CUDA().compute<PointXYZI>(
          input_cloud_tmp, LidarFrontLeft_Fine_Param[0], LidarFrontLeft_Fine_Param[1], LidarFrontLeft_Fine_Param[2],
          LidarFrontLeft_Fine_Param[3], LidarFrontLeft_Fine_Param[4], LidarFrontLeft_Fine_Param[5]);
      *input_cloud_tmp = CuboidFilter().hollow_removal<PointXYZI>(input_cloud_tmp, -7.0, 2, -1.3, 1.3, -3.0, 1);

      // ROI
      if (use_roi)
      {
        // *input_cloud_tmp = CuboidFilter().hollow_removal_IO<PointXYZI>(input_cloud_tmp, -7.0, 1, -1.4, 1.4, -3.0, 0.1,
        //                                                                -30, 4, 0, 30.0, -5.0, 0.01);
      }

      // assign
      *cloudPtr_LidarFrontLeft = *input_cloud_tmp;

      // publish
      cloudPtr_LidarFrontLeft->header.frame_id = "lidar";
      pub_LidarFrontLeft.publish(*cloudPtr_LidarFrontLeft);

      cout << "[L-Grabber]:" << stopWatch_L.getTimeSeconds() << 's' << endl;
    }
  }
  L_Lock.unlock();
}

void cloud_cb_LidarFrontRight(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  R_Lock.lock();
  if (input_cloud->width * input_cloud->height > 100)
  {
    stopWatch_R.reset();
    heartBeat[0] = true;

    // check data from hardware
    if (debug_output && (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) < 3600)
    {
      uint64_t diff_time = (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) * 1000;
      cout << "[Right->Gbr]: " << diff_time << "ms" << endl;
    }

    pcl::PointCloud<pcl::PointXYZIR>::Ptr input_cloud_tmp_ring(new pcl::PointCloud<pcl::PointXYZIR>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_tmp(new pcl::PointCloud<pcl::PointXYZI>);
    *input_cloud_tmp_ring = SensorMsgs_to_XYZIR(*input_cloud, "velodyne");    

    if (use_compress)
    {
      bool showStatistics = false;
      pcl::io::compression_Profiles_e compressionProfile = pcl::io::HIGH_RES_ONLINE_COMPRESSION_WITH_COLOR;
      pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA>* PointCloudEncoder;
      PointCloudEncoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA> (compressionProfile, showStatistics);

      // compressed stringstream
      msgs::CompressedPointCloud compressed_pointcloud;
      std::stringstream compressedData;

      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_xyzrgba(new pcl::PointCloud<pcl::PointXYZRGBA>);
      *cloud_xyzrgba = XYZIR_to_XYZRBGA(input_cloud_tmp_ring);
      
      PointCloudEncoder->encodePointCloud(cloud_xyzrgba, compressedData);
      compressed_pointcloud.data =  compressedData.str();

      compressed_pointcloud.header = input_cloud->header;

      pub_LidarFrontRight_Compress.publish(compressed_pointcloud);
      
      delete (PointCloudEncoder);

      compressedData.clear();
      compressedData.str("");

    }

    if (use_filter)
    {
      pcl::PointCloud<pcl::PointXYZIR>::Ptr output_cloud_tmp_ring(new pcl::PointCloud<pcl::PointXYZIR>);
      *output_cloud_tmp_ring = NoiseFilter().runRingOutlierRemoval(input_cloud_tmp_ring, 32, 0.3);
      pcl::copyPointCloud(*output_cloud_tmp_ring, *input_cloud_tmp);
    }
    else
    {
      pcl::copyPointCloud(*input_cloud_tmp_ring, *input_cloud_tmp);
    }

    // ------------------------------- check stitching
    if (GlobalVariable::STITCHING_MODE_NUM == 1)
    {
      *cloudPtr_LidarFrontRight = Transform_CUDA().compute<PointXYZI>(
          input_cloud_tmp, GlobalVariable::UI_PARA[6], GlobalVariable::UI_PARA[7], GlobalVariable::UI_PARA[8],
          GlobalVariable::UI_PARA[9], GlobalVariable::UI_PARA[10], GlobalVariable::UI_PARA[11]);

      cloudPtr_LidarFrontRight->header.frame_id = "lidar";
      pub_LidarFrontRight.publish(*cloudPtr_LidarFrontRight);
    }
    else
    {
      heartBeat[1] = false;

      // Transfrom
      *input_cloud_tmp = Transform_CUDA().compute<PointXYZI>(
          input_cloud_tmp, LidarFrontRight_Fine_Param[0], LidarFrontRight_Fine_Param[1], LidarFrontRight_Fine_Param[2],
          LidarFrontRight_Fine_Param[3], LidarFrontRight_Fine_Param[4], LidarFrontRight_Fine_Param[5]);
      *input_cloud_tmp = CuboidFilter().hollow_removal<PointXYZI>(input_cloud_tmp, -7.0, 2, -1.3, 1.3, -3.0, 1);
      
      // ROI
      if (use_roi)
      {
        // *input_cloud_tmp = CuboidFilter().hollow_removal_IO<PointXYZI>(input_cloud_tmp, -7.0, 1, -1.4, 1.4, -3.0, 0.1,
        //                                                                -30.0, 4, -30.0, 0, -5.0, 0.01);
      }

      // assign
      *cloudPtr_LidarFrontRight = *input_cloud_tmp;

      // publish
      cloudPtr_LidarFrontRight->header.frame_id = "lidar";
      pub_LidarFrontRight.publish(*cloudPtr_LidarFrontRight);
    }
    cout << "[R-Graber]: " << stopWatch_R.getTimeSeconds() << 's' << endl;
  }
  R_Lock.unlock();
}

void cloud_cb_LidarFrontTop(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  T_Lock.lock();
  if (input_cloud->width * input_cloud->height > 100)
  {
    stopWatch_T.reset();
    heartBeat[4] = true;

    // check data from hardware
    if (debug_output && (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) < 3600)
    {
      uint64_t diff_time = (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) * 1000;
      cout << "[Top->Gbr]: " << diff_time << "ms" << endl;
    }
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_tmp(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZIR>::Ptr input_cloud_tmp_ring(new pcl::PointCloud<pcl::PointXYZIR>);
    *input_cloud_tmp_ring = SensorMsgs_to_XYZIR(*input_cloud, "ouster"); 

    if (use_compress)
    {
      bool showStatistics = false;
      pcl::io::compression_Profiles_e compressionProfile = pcl::io::HIGH_RES_ONLINE_COMPRESSION_WITH_COLOR;
      pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA>* PointCloudEncoder;
      PointCloudEncoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBA> (compressionProfile, showStatistics);

      // compressed stringstream
      msgs::CompressedPointCloud compressed_pointcloud;
      std::stringstream compressedData;

      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_xyzrgba(new pcl::PointCloud<pcl::PointXYZRGBA>);
      *cloud_xyzrgba = XYZIR_to_XYZRBGA(input_cloud_tmp_ring);

      PointCloudEncoder->encodePointCloud(cloud_xyzrgba, compressedData);
      compressed_pointcloud.data =  compressedData.str();

      compressed_pointcloud.header = input_cloud->header;

      pub_LidarFrontTop_Compress.publish(compressed_pointcloud);
      
      delete (PointCloudEncoder);

      compressedData.clear();
      compressedData.str("");

    }

    // ring filter
    if (use_filter)
    {
      pcl::PointCloud<pcl::PointXYZIR>::Ptr output_cloud_tmp_ring(new pcl::PointCloud<pcl::PointXYZIR>);
      *output_cloud_tmp_ring = NoiseFilter().runRingOutlierRemoval(input_cloud_tmp_ring, 64, 1.5);
      pcl::copyPointCloud(*output_cloud_tmp_ring, *input_cloud_tmp);
    }
    else
    {
      pcl::copyPointCloud(*input_cloud_tmp_ring, *input_cloud_tmp);
    }

    // check stitching
    if (GlobalVariable::STITCHING_MODE_NUM == 1)
    {
      *cloudPtr_LidarFrontTop = Transform_CUDA().compute<PointXYZI>(input_cloud_tmp, 0, 0, 0, 0, 0.2, 0);

      cloudPtr_LidarFrontTop->header.frame_id = "lidar";
      pub_LidarFrontTop.publish(*cloudPtr_LidarFrontTop);
    }
    else
    {
      heartBeat[0] = false;
      
      // Transfrom
      *input_cloud_tmp = Transform_CUDA().compute<PointXYZI>(input_cloud_tmp, 0, 0, 0, 0, 0.2, 0);
      *input_cloud_tmp = CuboidFilter().hollow_removal<PointXYZI>(input_cloud_tmp, -7.0, 2, -1.3, 1.3, -3.0, 1);
      if (use_roi)
      {
        //*input_cloud_tmp = CuboidFilter().hollow_removal_IO<PointXYZI>(input_cloud_tmp, -7.0, 2, -1.2, 1.2, -3.0, 0.3,
                                                                      // -50, 50.0, -25.0, 25.0, -5.0, 0.01);
        //*ptr_cur_cloud = CuboidFilter().pass_through_soild<PointXYZI>(ptr_cur_cloud, -50, 50, -25, 25, -5, 1);
      }

      // assign
      stopWatch.reset();
      *cloudPtr_LidarFrontTop = *input_cloud_tmp;

      // publish
      cloudPtr_LidarFrontTop->header.frame_id = "lidar";
      pub_LidarFrontTop.publish(*cloudPtr_LidarFrontTop);
    }
    cout << "[T-Grabber]: "  <<  stopWatch_T.getTimeSeconds() << 's' << endl;
  }
  T_Lock.unlock();
}

void lidarAll_Pub(int lidarNum)
{
  //stopWatch.reset();
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

  // cloudPtr_LidarAll->header.stamp = ros::Time::now().toNSec() / 1000ull;

  LidarAll_time = cloudPtr_LidarAll->header.stamp;
  cloudPtr_LidarAll->header.frame_id = "lidar";

  //------ pub LidarAll
  if (cloudPtr_LidarAll->size() > 100)
  {
    pub_LidarAll.publish(*cloudPtr_LidarAll);
  }
  cloudPtr_LidarAll->clear();

  if (debug_output)
  {
    //cout << "[Grabber]: " << stopWatch.getTimeSeconds() << 's' << endl;
    // cout << "[Grabber]: " << ((ros::Time::now().toNSec()/1000ull) - cloudPtr_LidarFrontTop->header.stamp) / 1000000.0
    // << "s" << endl;
  }

  //------ clear real time memory
  // if wall_time - ros_time < 30 minutes, (not rosbag), clear sensor pc data memory if delay 2sec.
  uint64_t now = ros::Time::now().toNSec() / 1000ull;  // microsec
  if (!((now - LidarAll_time) > 1000000 * 1800))
  {
    if ((now - cloudPtr_LidarFrontLeft->header.stamp) > 1000000 * 1)
    {
      cloudPtr_LidarFrontLeft->clear();
      cout << "---------------------> Front-Left Clear" << endl;
    };
    if ((now - cloudPtr_LidarFrontRight->header.stamp) > 1000000 * 1)
    {
      cloudPtr_LidarFrontRight->clear();
      cout << "---------------------> Front-Right Clear" << endl;
    };
    if ((now - cloudPtr_LidarFrontTop->header.stamp) > 1000000 * 1)
    {
      cloudPtr_LidarFrontTop->clear();
      cout << "---------------------> Top Clear" << endl;
    };
  }
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
  ros::param::get("/use_compress", use_compress);
  ros::param::get("/use_roi", use_roi);

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

  // publisher
  pub_LidarFrontLeft = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarFrontLeft", 1);
  pub_LidarFrontRight = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarFrontRight", 1);
  pub_LidarFrontTop = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarFrontTop", 1);
  pub_LidarAll = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarAll", 1);

  // publisher - compressed
  pub_LidarFrontLeft_Compress = n.advertise<msgs::CompressedPointCloud>("/LidarFrontLeft/Compressed", 1);
  pub_LidarFrontRight_Compress = n.advertise<msgs::CompressedPointCloud>("/LidarFrontRight/Compressed", 1);
  pub_LidarFrontTop_Compress = n.advertise<msgs::CompressedPointCloud>("/LidarFrontTop/Compressed", 1);

  thread TheadDetection_UI(UI, argc, argv);
  thread TheadDetection_Pub(LidarAll_Publisher, argc, argv);

  ros::AsyncSpinner spinner(4);
  spinner.start();

  TheadDetection_UI.join();
  TheadDetection_Pub.join();
  ros::waitForShutdown();

  cout << "=============== Grabber Stop ===============" << endl;
  return 0;
}
