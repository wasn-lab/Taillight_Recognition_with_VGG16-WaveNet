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
#include <std_msgs/Empty.h>
#include "car_model.h"
#include "lidar_hardware.h"

//----------------------------- Stitching
vector<double> LidarFrontLeft_Fine_Param;
vector<double> LidarFrontRight_Fine_Param;
vector<double> Zero_Param(6, 0.0);

LiDARStitchingAuto LSA;

//---------------------------- pointcloud
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cloudPtr_LidarFrontLeft(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cloudPtr_LidarFrontRight(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cloudPtr_LidarFrontTop(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr g_cloudPtr_LidarAll(new pcl::PointCloud<pcl::PointXYZI>);

//---------------------------- Publisher
static ros::Publisher g_pub_LidarFrontLeft;
static ros::Publisher g_pub_LidarFrontRight;
static ros::Publisher g_pub_LidarFrontTop;
static ros::Publisher g_pub_LidarAll;

static ros::Publisher g_pub_LidarFrontLeft_Raw_HeartBeat;
static ros::Publisher g_pub_LidarFrontRight_Raw_HeartBeat;
static ros::Publisher g_pub_LidarFrontTop_Raw_HeartBeat;
static ros::Publisher g_pub_LidarAll_HeartBeat;

static ros::Publisher g_pub_LidarFrontTop_Localization;

static ros::Publisher g_pub_LidarFrontLeft_Compress;
static ros::Publisher g_pub_LidarFrontRight_Compress;
static ros::Publisher g_pub_LidarFrontTop_Compress;

//--------------------------- Global Variables
mutex g_L_Lock;
mutex g_R_Lock;
mutex g_T_Lock;

StopWatch g_stopWatch_L;
StopWatch g_stopWatch_R;
StopWatch g_stopWatch_T;
StopWatch g_stopWatch_Compressor;

bool g_debug_output = false;
bool g_use_filter = false;
bool g_use_oct_compress = false;
bool g_use_roi = false;

void lidarAll_Pub(int lidarNum);

uint64_t g_all_last_time = 0;

//------------------------ Compressor
void Compressor(pcl::PointCloud<pcl::PointXYZIR>::Ptr input_cloud_tmp_ring, ros::Publisher output_publisher);

//------------------------------ Callbacks
void cloud_cb_LidarFrontLeft(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  g_L_Lock.lock();

  // -------------------Raw/heartbeat publisher
  // check heartbeat by subcriber data receiver
  std_msgs::Empty empty_msg;
  g_pub_LidarFrontLeft_Raw_HeartBeat.publish(empty_msg);

  if (input_cloud->width * input_cloud->height > 100)
  {
    g_stopWatch_L.reset();

    // check data from hardware
    if (g_debug_output && (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) < 3600)
    {
      uint64_t diff_time = (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) * 1000;
      cout << "[Left->Gbr]: " << diff_time << "ms" << endl;
    }

    //-------------------------- sensor_msg to pcl XYZIR
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_tmp(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZIR>::Ptr input_cloud_tmp_ring(new pcl::PointCloud<pcl::PointXYZIR>);

#if CAR_MODEL_IS_B1_V2
    *input_cloud_tmp_ring = SensorMsgs_to_XYZIR(*input_cloud, lidar::Hardware::Velodyne);
#elif CAR_MODEL_IS_B1_V3
    *input_cloud_tmp_ring = SensorMsgs_to_XYZIR(*input_cloud, lidar::Hardware::Ouster);
#elif CAR_MODEL_IS_C1
    *input_cloud_tmp_ring = SensorMsgs_to_XYZIR(*input_cloud, lidar::Hardware::Ouster);
#else
    #error CORRESPONDING CAR MODEL NOT FOUND.
#endif

    //-------------------------- compress thread
    if (g_use_oct_compress)
    {
      thread t_LeftCompressor;
      t_LeftCompressor = thread{ Compressor, input_cloud_tmp_ring, g_pub_LidarFrontLeft_Compress };
      t_LeftCompressor.detach();
    }

    //-------------------------- ring filter
    if (g_use_filter)
    {
      pcl::PointCloud<pcl::PointXYZIR>::Ptr output_cloud_tmp_ring(new pcl::PointCloud<pcl::PointXYZIR>);
#if CAR_MODEL_IS_B1_V2
      *output_cloud_tmp_ring = NoiseFilter().runRingOutlierRemoval(input_cloud_tmp_ring, 32, 0.5);
#elif CAR_MODEL_IS_B1_V3
      *output_cloud_tmp_ring = NoiseFilter().runRingOutlierRemoval(input_cloud_tmp_ring, 64, 1.5);
#elif CAR_MODEL_IS_C1
      *output_cloud_tmp_ring = NoiseFilter().runRingOutlierRemoval(input_cloud_tmp_ring, 64, 1.5);
#else
      #error CORRESPONDING CAR MODEL NOT FOUND.
#endif
      pcl::copyPointCloud(*output_cloud_tmp_ring, *input_cloud_tmp);
    }
    else
    {
      pcl::copyPointCloud(*input_cloud_tmp_ring, *input_cloud_tmp);
    }

    //-------------------------- Transform
    if (GlobalVariable::STITCHING_MODE_NUM == 1)
    {
      *g_cloudPtr_LidarFrontLeft = Transform_CUDA().compute<PointXYZI>(
          input_cloud_tmp, GlobalVariable::UI_PARA[0], GlobalVariable::UI_PARA[1], GlobalVariable::UI_PARA[2],
          GlobalVariable::UI_PARA[3], GlobalVariable::UI_PARA[4], GlobalVariable::UI_PARA[5]);

      g_cloudPtr_LidarFrontLeft->header.frame_id = "lidar";
      g_pub_LidarFrontLeft.publish(*g_cloudPtr_LidarFrontLeft);
    }
    else
    {
      *input_cloud_tmp = Transform_CUDA().compute<PointXYZI>(
          input_cloud_tmp, LidarFrontLeft_Fine_Param[0], LidarFrontLeft_Fine_Param[1], LidarFrontLeft_Fine_Param[2],
          LidarFrontLeft_Fine_Param[3], LidarFrontLeft_Fine_Param[4], LidarFrontLeft_Fine_Param[5]);
      *input_cloud_tmp = CuboidFilter().hollow_removal<PointXYZI>(input_cloud_tmp, -7.0, 2, -1.3, 1.3, -3.0, 1);

      //-------------------------- ROI
      if (g_use_roi)
      {
        // *input_cloud_tmp = CuboidFilter().hollow_removal_IO<PointXYZI>(input_cloud_tmp, -7.0, 1, -1.4, 1.4, -3.0,
        // 0.1, -30, 4, 0, 30.0, -5.0, 0.01);
      }

      // assign
      *g_cloudPtr_LidarFrontLeft = *input_cloud_tmp;

      // publish
      g_cloudPtr_LidarFrontLeft->header.frame_id = "lidar";
      g_pub_LidarFrontLeft.publish(*g_cloudPtr_LidarFrontLeft);

      cout << "[L-Gbr]: " << g_stopWatch_L.getTimeSeconds() << 's' << endl;
    }
  }
  g_L_Lock.unlock();
}

void cloud_cb_LidarFrontRight(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  g_R_Lock.lock();

  // -------------------Raw/heartbeat publisher
  // check heartbeat by subcriber data receiver
  std_msgs::Empty empty_msg;
  g_pub_LidarFrontRight_Raw_HeartBeat.publish(empty_msg);

  if (input_cloud->width * input_cloud->height > 100)
  {
    g_stopWatch_R.reset();

    // check data from hardware
    if (g_debug_output && (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) < 3600)
    {
      uint64_t diff_time = (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) * 1000;
      cout << "[Right->Gbr]: " << diff_time << "ms" << endl;
    }

    pcl::PointCloud<pcl::PointXYZIR>::Ptr input_cloud_tmp_ring(new pcl::PointCloud<pcl::PointXYZIR>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_tmp(new pcl::PointCloud<pcl::PointXYZI>);
#if CAR_MODEL_IS_B1_V2
    *input_cloud_tmp_ring = SensorMsgs_to_XYZIR(*input_cloud, lidar::Hardware::Velodyne);
#elif CAR_MODEL_IS_B1_V3
    *input_cloud_tmp_ring = SensorMsgs_to_XYZIR(*input_cloud, lidar::Hardware::Ouster);
#elif CAR_MODEL_IS_C1
    *input_cloud_tmp_ring = SensorMsgs_to_XYZIR(*input_cloud, lidar::Hardware::Ouster);
#else
    #error CORRESPONDING CAR MODEL NOT FOUND.
#endif

    if (g_use_oct_compress)
    {
      thread t_RightCompressor;
      t_RightCompressor = thread{ Compressor, input_cloud_tmp_ring, g_pub_LidarFrontRight_Compress };
      t_RightCompressor.detach();
    }

    if (g_use_filter)
    {
      pcl::PointCloud<pcl::PointXYZIR>::Ptr output_cloud_tmp_ring(new pcl::PointCloud<pcl::PointXYZIR>);
#if CAR_MODEL_IS_B1_V2
      *output_cloud_tmp_ring = NoiseFilter().runRingOutlierRemoval(input_cloud_tmp_ring, 32, 0.5);
#elif CAR_MODEL_IS_B1_V3
      *output_cloud_tmp_ring = NoiseFilter().runRingOutlierRemoval(input_cloud_tmp_ring, 64, 1.5);
#elif CAR_MODEL_IS_C1
      *output_cloud_tmp_ring = NoiseFilter().runRingOutlierRemoval(input_cloud_tmp_ring, 64, 1.5);
#else
    #error CORRESPONDING CAR MODEL NOT FOUND.
#endif
      pcl::copyPointCloud(*output_cloud_tmp_ring, *input_cloud_tmp);
    }
    else
    {
      pcl::copyPointCloud(*input_cloud_tmp_ring, *input_cloud_tmp);
    }

    // ------------------------------- check stitching
    if (GlobalVariable::STITCHING_MODE_NUM == 1)
    {
      *g_cloudPtr_LidarFrontRight = Transform_CUDA().compute<PointXYZI>(
          input_cloud_tmp, GlobalVariable::UI_PARA[6], GlobalVariable::UI_PARA[7], GlobalVariable::UI_PARA[8],
          GlobalVariable::UI_PARA[9], GlobalVariable::UI_PARA[10], GlobalVariable::UI_PARA[11]);

      g_cloudPtr_LidarFrontRight->header.frame_id = "lidar";
      g_pub_LidarFrontRight.publish(*g_cloudPtr_LidarFrontRight);
    }
    else
    {
      // Transfrom
      *input_cloud_tmp = Transform_CUDA().compute<PointXYZI>(
          input_cloud_tmp, LidarFrontRight_Fine_Param[0], LidarFrontRight_Fine_Param[1], LidarFrontRight_Fine_Param[2],
          LidarFrontRight_Fine_Param[3], LidarFrontRight_Fine_Param[4], LidarFrontRight_Fine_Param[5]);
      *input_cloud_tmp = CuboidFilter().hollow_removal<PointXYZI>(input_cloud_tmp, -7.0, 2, -1.3, 1.3, -3.0, 1);

      // ROI
      if (g_use_roi)
      {
        // *input_cloud_tmp = CuboidFilter().hollow_removal_IO<PointXYZI>(input_cloud_tmp, -7.0, 1, -1.4, 1.4, -3.0,
        // 0.1, -30.0, 4, -30.0, 0, -5.0, 0.01);
      }

      // assign
      *g_cloudPtr_LidarFrontRight = *input_cloud_tmp;

      // publish
      g_cloudPtr_LidarFrontRight->header.frame_id = "lidar";
      g_pub_LidarFrontRight.publish(*g_cloudPtr_LidarFrontRight);
    }
    cout << "[R-Gbr]: " << g_stopWatch_R.getTimeSeconds() << 's' << endl;
  }
  g_R_Lock.unlock();
}

void cloud_cb_LidarFrontTop(const boost::shared_ptr<const sensor_msgs::PointCloud2>& input_cloud)
{
  g_T_Lock.lock();

  // -------------------Raw/heartbeat publisher
  // check heartbeat by subcriber data receiver
  std_msgs::Empty empty_msg;
  g_pub_LidarFrontTop_Raw_HeartBeat.publish(empty_msg);

  if (input_cloud->width * input_cloud->height > 100)
  {
    g_stopWatch_T.reset();

    // check data from hardware
    if (g_debug_output && (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) < 3600)
    {
      uint64_t diff_time = (ros::Time::now().toSec() - input_cloud->header.stamp.toSec()) * 1000;
      cout << "[Top->Gbr]: " << diff_time << "ms" << endl;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud_tmp(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZIR>::Ptr input_cloud_tmp_ring(new pcl::PointCloud<pcl::PointXYZIR>);
    *input_cloud_tmp_ring = SensorMsgs_to_XYZIR(*input_cloud, lidar::Hardware::Ouster);

    if (g_use_oct_compress)
    {
      thread t_TopCompressor;
      t_TopCompressor = thread{ Compressor, input_cloud_tmp_ring, g_pub_LidarFrontTop_Compress };
      t_TopCompressor.detach();
    }

    //------------------- For Localization
    pcl::PointCloud<pcl::PointXYZI>::Ptr localization_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*input_cloud_tmp_ring, *localization_cloud);
#if CAR_MODEL_IS_B1_V2 || CAR_MODEL_IS_B1_V3
    *localization_cloud = Transform_CUDA().compute<PointXYZI>(localization_cloud, 0, 0, 0, 0, 0.2, 0);
#elif CAR_MODEL_IS_C1
    *localization_cloud = Transform_CUDA().compute<PointXYZI>(localization_cloud, 0, 0, 0, 0.023, 0.21, 0);
#else
    #error CORRESPONDING CAR MODEL NOT FOUND.
#endif

    g_pub_LidarFrontTop_Localization.publish(*localization_cloud);

    // Ring Filter
    if (g_use_filter)
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
#if CAR_MODEL_IS_B1_V2 || CAR_MODEL_IS_B1_V3
      *g_cloudPtr_LidarFrontTop = Transform_CUDA().compute<PointXYZI>(input_cloud_tmp, 0, 0, 0, 0, 0.2, 0);
#elif CAR_MODEL_IS_C1
      *g_cloudPtr_LidarFrontTop = Transform_CUDA().compute<PointXYZI>(input_cloud_tmp, 0, 0, 0, 0.023, 0.21, 0);
#else
      #error CORRESPONDING CAR MODEL NOT FOUND.
#endif
      g_cloudPtr_LidarFrontTop->header.frame_id = "lidar";
      g_pub_LidarFrontTop.publish(*g_cloudPtr_LidarFrontTop);
    }
    else
    {
      // Transfrom
#if CAR_MODEL_IS_B1_V2 || CAR_MODEL_IS_B1_V3
      *input_cloud_tmp = Transform_CUDA().compute<PointXYZI>(input_cloud_tmp, 0, 0, 0, 0, 0.2, 0);
#elif CAR_MODEL_IS_C1
      *input_cloud_tmp = Transform_CUDA().compute<PointXYZI>(input_cloud_tmp, 0, 0, 0, 0.023, 0.21, 0);
#else
      #error CORRESPONDING CAR MODEL NOT FOUND.
#endif
      *input_cloud_tmp = CuboidFilter().hollow_removal<PointXYZI>(input_cloud_tmp, -7.0, 2, -1.3, 1.3, -3.0, 1);
      if (g_use_roi)
      {
        //*input_cloud_tmp = CuboidFilter().hollow_removal_IO<PointXYZI>(input_cloud_tmp, -7.0, 2, -1.2, 1.2, -3.0, 0.3,
        // -50, 50.0, -25.0, 25.0, -5.0, 0.01);
        //*ptr_cur_cloud = CuboidFilter().pass_through_soild<PointXYZI>(ptr_cur_cloud, -50, 50, -25, 25, -5, 1);
      }

      // assign
      *g_cloudPtr_LidarFrontTop = *input_cloud_tmp;

      // publish
      g_cloudPtr_LidarFrontTop->header.frame_id = "lidar";
      g_pub_LidarFrontTop.publish(*g_cloudPtr_LidarFrontTop);
    }
    cout << "[T-Gbr]: " << g_stopWatch_T.getTimeSeconds() << 's' << endl;
  }
  g_T_Lock.unlock();
}

//---------------------------------------------------- Point Cloud Compression Thread
void Compressor(pcl::PointCloud<pcl::PointXYZIR>::Ptr input_cloud_tmp_ring, ros::Publisher output_publisher)
{
  mutex Compressor_Lock;
  Compressor_Lock.lock();
  g_stopWatch_Compressor.reset();

  //--------------------------- create compressor
  pcl::io::compression_Profiles_e compressionProfile = pcl::io::MANUAL_CONFIGURATION;
  bool showStatistics = false;
  const double pointResolution_arg = 0.001;
  const double octreeResolution_arg = 0.001;
  bool doVoxelGridDownDownSampling_arg = false;
  const unsigned int iFrameRate_arg = 30;
  bool doColorEncoding_arg = true;
  const unsigned char colorBitResolution_arg = 8;

  pcl::io::OctreePointCloudCompression<pcl::PointXYZRGB>* PointCloudEncoder;
  PointCloudEncoder =
      // new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGB>(compressionProfile, showStatistics);
      new pcl::io::OctreePointCloudCompression<pcl::PointXYZRGB>(
          compressionProfile, showStatistics, pointResolution_arg, octreeResolution_arg,
          doVoxelGridDownDownSampling_arg, iFrameRate_arg, doColorEncoding_arg, colorBitResolution_arg);

  // compressed stringstream
  msgs::CompressedPointCloud compressed_pointcloud;
  std::stringstream compressedData;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  *cloud_xyzrgb = XYZIR_to_XYZRGB(input_cloud_tmp_ring);

  PointCloudEncoder->encodePointCloud(cloud_xyzrgb, compressedData);
  compressed_pointcloud.data = compressedData.str();

  pcl_conversions::fromPCL(input_cloud_tmp_ring->header, compressed_pointcloud.header);

  output_publisher.publish(compressed_pointcloud);

  delete (PointCloudEncoder);

  compressedData.str("");
  compressedData.clear();
  //------------------------- end of compress

  if (g_stopWatch_Compressor.getTimeSeconds() > 0.05)
  {
    cout << "COMPRESSION DELAY ----------> " << g_stopWatch_Compressor.getTimeSeconds() << 's' << endl;
  }
  Compressor_Lock.unlock();
}

//----------------------------------------------------- Publisher
void LidarAll_Publisher(int argc, char** argv)
{
  ros::Rate loop_rate(20);
  while (ros::ok())
  {
    lidarAll_Pub(4);
    loop_rate.sleep();
  }
}

void lidarAll_Pub(int lidarNum)
{
  //------ combine pointcloud
  *g_cloudPtr_LidarAll += *g_cloudPtr_LidarFrontLeft;
  *g_cloudPtr_LidarAll += *g_cloudPtr_LidarFrontRight;
  *g_cloudPtr_LidarAll += *g_cloudPtr_LidarFrontTop;

  //------ assign header
  vector<uint64_t> stamp_vec;
  stamp_vec = { g_cloudPtr_LidarFrontTop->header.stamp, g_cloudPtr_LidarFrontLeft->header.stamp,
                g_cloudPtr_LidarFrontRight->header.stamp };
  uint64_t biggest_stamp = *max_element(std::begin(stamp_vec), std::end(stamp_vec));

  g_cloudPtr_LidarAll->header.stamp = biggest_stamp;
  g_cloudPtr_LidarAll->header.frame_id = "lidar";

  // if (g_cloudPtr_LidarFrontTop->header.stamp != 0)
  // {
  //   g_cloudPtr_LidarAll->header.stamp = g_cloudPtr_LidarFrontTop->header.stamp;
  // }
  // else
  // {
  //   if (g_cloudPtr_LidarFrontLeft->header.stamp >= g_cloudPtr_LidarFrontRight->header.stamp)
  //   {
  //     g_cloudPtr_LidarAll->header.stamp = g_cloudPtr_LidarFrontLeft->header.stamp;
  //   }
  //   else
  //   {
  //     g_cloudPtr_LidarAll->header.stamp = g_cloudPtr_LidarFrontRight->header.stamp;
  //   }
  // }

  // g_cloudPtr_LidarAll->header.stamp = ros::Time::now().toNSec() / 1000ull;

  //------ pub LidarAll
  if (g_cloudPtr_LidarAll->size() > 100 && g_cloudPtr_LidarAll->header.stamp != g_all_last_time)
  {
    g_pub_LidarAll.publish(*g_cloudPtr_LidarAll);
    std_msgs::Empty empty_msg;
    g_pub_LidarAll_HeartBeat.publish(empty_msg);
    g_all_last_time = g_cloudPtr_LidarAll->header.stamp;
  }
  g_cloudPtr_LidarAll->clear();

  //------ clear real time memory
  // if wall_time - ros_time < 30 minutes, (not rosbag), clear sensor pc data memory if delay 2sec.
  uint64_t now = ros::Time::now().toNSec() / 1000ull;  // microsec
  if (!((now - g_cloudPtr_LidarAll->header.stamp) > 1000000 * 1800))
  {
    if ((now - g_cloudPtr_LidarFrontLeft->header.stamp) > 1000000 * 1)
    {
      g_cloudPtr_LidarFrontLeft->clear();
      cout << "---------------------> Front-Left Clear" << endl;
    };
    if ((now - g_cloudPtr_LidarFrontRight->header.stamp) > 1000000 * 1)
    {
      g_cloudPtr_LidarFrontRight->clear();
      cout << "---------------------> Front-Right Clear" << endl;
    };
    if ((now - g_cloudPtr_LidarFrontTop->header.stamp) > 1000000 * 1)
    {
      g_cloudPtr_LidarFrontTop->clear();
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

int main(int argc, char** argv)
{
  cout << "=============== Grabber Start ===============" << endl;

  ros::init(argc, argv, "lidars_grabber");
  ros::NodeHandle n;

  // check debug mode
  ros::param::get("/debug_output", g_debug_output);
  ros::param::get("/use_filter", g_use_filter);
  ros::param::get("/use_oct_compress", g_use_oct_compress);
  ros::param::get("/use_roi", g_use_roi);

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

  ROS_INFO("Wait for /LidarFrontTop/Raw");
  ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/LidarFrontTop/Raw");
  ROS_INFO("Wait for /LidarFrontLeft/Raw");
  ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/LidarFrontLeft/Raw");
  ROS_INFO("Wait for /LidarFrontRight/Raw");
  ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/LidarFrontRight/Raw");

  ROS_INFO("/LidarFrontTop/Raw, /LidarFrontLeft/Raw, /LidarFrontRight/Raw are ready.");

  // subscriber
  ros::Subscriber sub_LidarFrontLeft =
      n.subscribe<sensor_msgs::PointCloud2>("/LidarFrontLeft/Raw", 1, cloud_cb_LidarFrontLeft);
  ros::Subscriber sub_LidarFrontRight =
      n.subscribe<sensor_msgs::PointCloud2>("/LidarFrontRight/Raw", 1, cloud_cb_LidarFrontRight);
  ros::Subscriber sub_LidarFrontTop =
      n.subscribe<sensor_msgs::PointCloud2>("/LidarFrontTop/Raw", 1, cloud_cb_LidarFrontTop);

  // publisher
  g_pub_LidarFrontLeft = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarFrontLeft", 1);
  g_pub_LidarFrontRight = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarFrontRight", 1);
  g_pub_LidarFrontTop = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarFrontTop", 1);
  g_pub_LidarAll = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarAll", 1);

  // publisher - heartbeat
  g_pub_LidarFrontLeft_Raw_HeartBeat = n.advertise<std_msgs::Empty>("/LidarFrontLeft/Raw/heartbeat", 1);
  g_pub_LidarFrontRight_Raw_HeartBeat = n.advertise<std_msgs::Empty>("/LidarFrontRight/Raw/heartbeat", 1);
  g_pub_LidarFrontTop_Raw_HeartBeat = n.advertise<std_msgs::Empty>("/LidarFrontTop/Raw/heartbeat", 1);
  g_pub_LidarAll_HeartBeat = n.advertise<std_msgs::Empty>("/LidarAll/heartbeat", 1);

  // publisher - localization
  g_pub_LidarFrontTop_Localization = n.advertise<pcl::PointCloud<pcl::PointXYZI> >("/LidarFrontTop/Localization", 1);

  // publisher - compressed
  g_pub_LidarFrontLeft_Compress = n.advertise<msgs::CompressedPointCloud>("/LidarFrontLeft/Oct_Compressed", 1);
  g_pub_LidarFrontRight_Compress = n.advertise<msgs::CompressedPointCloud>("/LidarFrontRight/Oct_Compressed", 1);
  g_pub_LidarFrontTop_Compress = n.advertise<msgs::CompressedPointCloud>("/LidarFrontTop/Oct_Compressed", 1);

  thread ThreadDetection_UI(UI, argc, argv);
  thread ThreadDetection_Pub(LidarAll_Publisher, argc, argv);

  ros::AsyncSpinner spinner(4);
  spinner.start();

  ThreadDetection_UI.join();
  ThreadDetection_Pub.join();

  ros::waitForShutdown();

  cout << "=============== Grabber Stop ===============" << endl;
  return 0;
}
