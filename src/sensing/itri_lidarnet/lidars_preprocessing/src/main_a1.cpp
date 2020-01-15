#include "all_header.h"
#include "GlobalVariable.h"
#include "debug_tool.h"
#include "KeyboardMouseEvent.h"
#include "UI/QtViewer.h"
#include "VoxelGrid_CUDA.h"
#include "VoxelFilter_CUDA.h"
#include "LidarToCamera.hpp"
#include "RayGroundFilter.h"
#include "CuboidFilter.h"
#include "PlaneGroundFilter.h"
#include "S1Cluster/S1Cluster.h"
#include "S2Track/S2Track.h"
#include "S3Classify/S3Classify.h"
#include "RosModuleA.hpp"
#include "Transmission/UdpClientServer.h"
#include "project_to_sphere_image.hpp"
#include "CompressFunction.h"
#include "Transmission/CanModuleA.hpp"

boost::mutex mutex_LidFrontTop;
boost::mutex mutex_LidFrontLeft;
boost::mutex mutex_LidFrontRight;
boost::mutex mutex_LidRearLeft;
boost::mutex mutex_LidRearRight;

pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidFrontTop (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidFrontLeft (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidFrontRight (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidRearLeft (new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr_LidRearRight (new pcl::PointCloud<pcl::PointXYZI>);

ros::Time lidar_time[5];

void
callback_LidFrontTop (const msgs::PointCloud::ConstPtr& msg)
{
  //cout << "LidFrontTop  seq: " << msg->pclHeader.seq << ", ROS_time : " << msg->pclHeader.stamp << " sec" << endl;
  mutex_LidFrontTop.lock ();

  lidar_time[0] = msg->lidHeader.stamp;

  cloudPtr_LidFrontTop->width = msg->pointCloud.size ();
  cloudPtr_LidFrontTop->height = 1;
  cloudPtr_LidFrontTop->is_dense = false;
  cloudPtr_LidFrontTop->points.resize (msg->pointCloud.size ());

#pragma omp parallel for
  for (size_t i = 0; i < msg->pointCloud.size (); ++i)
  {
    cloudPtr_LidFrontTop->points[i].x = msg->pointCloud[i].x;
    cloudPtr_LidFrontTop->points[i].y = msg->pointCloud[i].y;
    cloudPtr_LidFrontTop->points[i].z = msg->pointCloud[i].z;
    //cloudPtr_LidFrontTop->points[i].intensity = (int) msg->pointCloud[i].intensity;  //(int) msg->pointCloud[i].intensity
    cloudPtr_LidFrontTop->points[i].intensity = 0;
  }

  mutex_LidFrontTop.unlock ();
}

void
callback_LidFrontLeft (const msgs::PointCloud::ConstPtr& msg)
{
  mutex_LidFrontLeft.lock ();

  lidar_time[1] = msg->lidHeader.stamp;

  cloudPtr_LidFrontLeft->width = msg->pointCloud.size ();
  cloudPtr_LidFrontLeft->height = 1;
  cloudPtr_LidFrontLeft->is_dense = false;
  cloudPtr_LidFrontLeft->points.resize (msg->pointCloud.size ());

#pragma omp parallel for
  for (size_t i = 0; i < msg->pointCloud.size (); ++i)
  {
    cloudPtr_LidFrontLeft->points[i].x = msg->pointCloud[i].x;
    cloudPtr_LidFrontLeft->points[i].y = msg->pointCloud[i].y;
    cloudPtr_LidFrontLeft->points[i].z = msg->pointCloud[i].z;
    //cloudPtr_LidFrontLeft->points[i].intensity = (int) msg->pointCloud[i].intensity;  //(int) msg->pointCloud[i].intensity
    cloudPtr_LidFrontLeft->points[i].intensity = 0;
  }

  mutex_LidFrontLeft.unlock ();
}

void
callback_LidFrontRight (const msgs::PointCloud::ConstPtr& msg)
{
  mutex_LidFrontRight.lock ();

  lidar_time[2] = msg->lidHeader.stamp;

  cloudPtr_LidFrontRight->width = msg->pointCloud.size ();
  cloudPtr_LidFrontRight->height = 1;
  cloudPtr_LidFrontRight->is_dense = false;
  cloudPtr_LidFrontRight->points.resize (msg->pointCloud.size ());

#pragma omp parallel for
  for (size_t i = 0; i < msg->pointCloud.size (); ++i)
  {
    cloudPtr_LidFrontRight->points[i].x = msg->pointCloud[i].x;
    cloudPtr_LidFrontRight->points[i].y = msg->pointCloud[i].y;
    cloudPtr_LidFrontRight->points[i].z = msg->pointCloud[i].z;
    //cloudPtr_LidFrontRight->points[i].intensity = (int) msg->pointCloud[i].intensity;  //(int) msg->pointCloud[i].intensity
    cloudPtr_LidFrontRight->points[i].intensity = 0;
  }

  mutex_LidFrontRight.unlock ();
}

void
callback_LidRearLeft (const msgs::PointCloud::ConstPtr& msg)
{
  mutex_LidRearLeft.lock ();

  lidar_time[3] = msg->lidHeader.stamp;

  cloudPtr_LidRearLeft->width = msg->pointCloud.size ();
  cloudPtr_LidRearLeft->height = 1;
  cloudPtr_LidRearLeft->is_dense = false;
  cloudPtr_LidRearLeft->points.resize (msg->pointCloud.size ());

#pragma omp parallel for
  for (size_t i = 0; i < msg->pointCloud.size (); ++i)
  {
    cloudPtr_LidRearLeft->points[i].x = msg->pointCloud[i].x;
    cloudPtr_LidRearLeft->points[i].y = msg->pointCloud[i].y;
    cloudPtr_LidRearLeft->points[i].z = msg->pointCloud[i].z;
    //cloudPtr_LidRearLeft->points[i].intensity = (int) msg->pointCloud[i].intensity;  //(int) msg->pointCloud[i].intensity
    cloudPtr_LidRearLeft->points[i].intensity = 0;
  }

  mutex_LidRearLeft.unlock ();
}

void
callback_LidRearRight (const msgs::PointCloud::ConstPtr& msg)
{
  mutex_LidRearRight.lock ();

  lidar_time[4] = msg->lidHeader.stamp;

  cloudPtr_LidRearRight->width = msg->pointCloud.size ();
  cloudPtr_LidRearRight->height = 1;
  cloudPtr_LidRearRight->is_dense = false;
  cloudPtr_LidRearRight->points.resize (msg->pointCloud.size ());

#pragma omp parallel for
  for (size_t i = 0; i < msg->pointCloud.size (); ++i)
  {
    cloudPtr_LidRearRight->points[i].x = msg->pointCloud[i].x;
    cloudPtr_LidRearRight->points[i].y = msg->pointCloud[i].y;
    cloudPtr_LidRearRight->points[i].z = msg->pointCloud[i].z;
    //cloudPtr_LidRearRight->points[i].intensity = (int) msg->pointCloud[i].intensity;  //(int) msg->pointCloud[i].intensity
    cloudPtr_LidRearRight->points[i].intensity = 0;  //(int) msg->pointCloud[i].intensity
  }

  mutex_LidRearRight.unlock ();
}

class ObjectsDetectionA
{
  private:

    thread mThread;
    PointCloud<PointXYZI>::Ptr release_cloud;
    StopWatch timer_algorithm_running;
    S1Cluster S1cluster;
    //S2Track S2track;
    //S3Classify S3classify;
    //LidarToCamera ToCameraMiddles;
    //UdpClient UDPclient;

  public:
    ObjectsDetectionA ()
    {
    }

    void
    initial (boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer,
             int *input_viewID)
    {
      timer_algorithm_running.reset ();
      S1cluster = S1Cluster (input_viewer, input_viewID);
      //S2track = S2Track (input_viewer, input_viewID);
      //S3classify = S3Classify (input_viewer, input_viewID);
      //ToCameraMiddles = LidarToCamera (input_viewer, input_viewID, 1250, -4.9 * D2R, -2.1 * D2R, -0.3 * D2R, -60, -0.5, -1340, 1280, 720);
      //UDPclient.initial (GlobalVariable::UI_UDP_IP, GlobalVariable::UI_UDP_Port);
    }

    void
    setInputCloud (PointCloud<PointXYZI>::Ptr input)
    {
      release_cloud = input;
    }

    void
    startThread ()
    {
      mThread = thread (&ObjectsDetectionA::update, this);
    }

    void
    waitThread ()
    {
      if (mThread.joinable ())
        mThread.join ();
    }

    void
    update ()
    {
      timer_algorithm_running.reset ();

      PointCloud<PointXYZ>::Ptr ptr_cur_cloud (new PointCloud<PointXYZ>);
      pcl::copyPointCloud (*release_cloud, *ptr_cur_cloud);
      ptr_cur_cloud->width = release_cloud->size ();
      ptr_cur_cloud->height = 1;
      ptr_cur_cloud->is_dense = false;
      ptr_cur_cloud->points.resize (ptr_cur_cloud->width);
      //cout << "[raw data       ]:" << ptr_cur_cloud->size () << endl;

      KeyboardMouseEvent ().setCloudToPCD (*ptr_cur_cloud);

//      *ptr_cur_cloud = remove_ground (ptr_cur_cloud, -0.000490628, -0.0249136, 0.999689, 0.723542);
//      cout << "[remove_plane]:" << ptr_cur_cloud->size () << "," << timer_algorithm_running.getTimeSeconds () << "s" << endl;
//
//      pcl::PointIndicesPtr indices_ground (new pcl::PointIndices);
//      *indices_ground = remove_ground_sample_consensus_model (ptr_cur_cloud);
//      PointCloud<PointXYZ>::Ptr cloud_ground (new PointCloud<PointXYZ>);
//      extract_Indices (ptr_cur_cloud, indices_ground, *cloud_ground, *ptr_cur_cloud);
//      cout << "[remove ground SCM ]:" << ptr_cur_cloud->size () << "," << timer_algorithm_running.getTimeSeconds () << "s" << endl;
//
//      pcl::PointIndicesPtr indices_ground (new pcl::PointIndices);
//      *indices_ground = remove_ground_sac_segmentation (ptr_cur_cloud);
//      PointCloud<PointXYZ>::Ptr cloud_ground (new PointCloud<PointXYZ>);
//      extract_Indices (ptr_cur_cloud, indices_ground, *cloud_ground, *ptr_cur_cloud);
//      cout << "[remove ground SAC ]:" << ptr_cur_cloud->size () << "," << timer_algorithm_running.getTimeSeconds () << "s" << endl;
//
//      vector<PointCloud<PointXYZ>> roi_cloud;
//      roi_cloud = separate_cloud (ptr_cur_cloud, -GlobalVariable::UI_LATERAL_RANGE, GlobalVariable::UI_LATERAL_RANGE, 0, 50, -5, -2.2);
//      PointCloud<PointXYZ>::Ptr cloud_roi_inside (new PointCloud<PointXYZ>);
//      PointCloud<PointXYZ>::Ptr cloud_roi_outside (new PointCloud<PointXYZ>);
//      *cloud_roi_inside = roi_cloud.at (0);
//      *cloud_roi_outside = roi_cloud.at (1);
//      cout << "[get ground ROI    ]:" << timer_algorithm_running.getTimeSeconds () << "s" << endl;
//
//      *ptr_cur_cloud = uniform_sampling (ptr_cur_cloud, GlobalVariable::UI_UNIFORM_SAMPLING);
//      cout << "[down sampling  ]:" << ptr_cur_cloud->size () << "," << timer_algorithm_running.getTimeSeconds () << "s" << endl;

      *ptr_cur_cloud = CuboidFilter ().pass_through_soild<PointXYZ> (ptr_cur_cloud, -12.5, 50, -10, 12, -5, 0);
      *ptr_cur_cloud = CuboidFilter ().hollow_removal<PointXYZ> (ptr_cur_cloud, -6.6, 0.9, -1.45, 1.45, -5, 0);
      //cout << "[pass through   ]:" << ptr_cur_cloud->size () << "," << timer_algorithm_running.getTimeSeconds () << "s" << endl;

      *ptr_cur_cloud = VoxelGrid_CUDA ().compute (ptr_cur_cloud, 0.2);
      //cout << "[voxel_grid     ]:" << ptr_cur_cloud->size () << "," << timer_algorithm_running.getTimeSeconds () << "s" << endl;

      pcl::PointIndicesPtr indices_ground (new pcl::PointIndices);
      //*indices_ground = remove_ground_approximate_progressive_morphological (ptr_cur_cloud);
      *indices_ground = RayGroundFilter (2.68, 3.0, 37.0, 0.1, 0.0001, 0.37, 0.3, 0.0001, 0.0001).compute (ptr_cur_cloud);
      PointCloud<PointXYZ>::Ptr cloud_ground (new PointCloud<PointXYZ>);
      PointCloud<PointXYZ>::Ptr cloud_non_ground (new PointCloud<PointXYZ>);
      extract_Indices (ptr_cur_cloud, indices_ground, *cloud_ground, *cloud_non_ground);
      //cout << "[remove ground]:" << timer_algorithm_running.getTimeSeconds () << "s"<< endl;

      if (cloud_ground->size () < 100)
      {
        RosModuleA::send_ErrorCode (0x4000);
        cout << "error: not find ground" << endl;
      }

      //vector<PointCloud<PointXYZ>> roi_cloud = separate_cloud (cloud_non_ground, 0.65, 50, -2, 2, -5, 0);
      //PointCloud<PointXYZ>::Ptr roi_buffer (new PointCloud<PointXYZ>);
      //*roi_buffer = roi_cloud.at (1);
      //*ptr_cur_cloud = NoiseFilter_CUDA ().compute (roi_buffer, 1.5, 1) + roi_cloud.at (0);

      *ptr_cur_cloud = VoxelFilter_CUDA ().compute (cloud_non_ground, 1, 1);
      //*ptr_cur_cloud = radius_outlier_removal (roi_buffer, 1, 1);
      //cout << "[remove outlier]:" << ptr_cur_cloud->size () << "," << timer_algorithm_running.getTimeSeconds () << "s" << endl;

      //tool_get_plane_coefficient (ptr_cur_cloud);

      //if (GlobalVariable::ENABLE_DEBUG_MODE) show_cloud(ptr_cur_cloud);

      int cur_cluster_num = 0;

      *cloud_ground = CuboidFilter ().pass_through_soild<PointXYZ> (cloud_ground, 0.65, 5, -2, 2, -5, 1);

      pcl::ModelCoefficients MC;
      MC = PlaneGroundFilter ().getCoefficientsSAC (cloud_ground, 2.68);  //baby car 0.65 luxgen MVP7 0.723, ITRI bus 2.68
      //float dis = pointToPlaneDistance (PointXYZ (0, 0, 0), MC.values[0], MC.values[1], MC.values[2], MC.values[3]);
      //cout << "[Sensor to ground  ]:" << timer_algorithm_running.getTimeSeconds () << "s " << dis << "m" << endl;

      S1cluster.setPlaneParameter (MC);
      CLUSTER_INFO* cur_cluster = S1cluster.getClusters (GlobalVariable::ENABLE_DEBUG_MODE, ptr_cur_cloud, &cur_cluster_num);
      //cout << "[S1cluster         ]:" << timer_algorithm_running.getTimeSeconds () << "s" << endl;

      //S2track.update (GlobalVariable::DEBUG_MODE, cur_cluster, cur_cluster_num);
      //cout << "[S2track           ]:" << timer_algorithm_running.getTimeSeconds () << "s" << endl;

      //S3classify.update (GlobalVariable::ENABLE_DEBUG_MODE, cur_cluster, cur_cluster_num);
      //cout << "[S3classify        ]:" << timer_algorithm_running.getTimeSeconds () << "s" << endl;

      //ToCameraMiddles.update (GlobalVariable::DEBUG_MODE, cur_cluster, cur_cluster_num);
      //cout << "[3D to 2D          ]:" << timer_algorithm_running.getTimeSeconds () << "s" << endl;

      RosModuleA::send_LidFus (cur_cluster, cur_cluster_num);
      //RosModuleA::send_LidObj (cur_cluster, cur_cluster_num);
      RosModuleA::send_LidRoi (cur_cluster, cur_cluster_num);

      //CanModule::ReadAndWrite_controller (cur_cluster, cur_cluster_num);

      //int sendnumber = UDPclient.send_obj_to_vcu (cur_cluster, cur_cluster_num);
      //int sendnumber = UDPclient.send_obj_to_server (cur_cluster, cur_cluster_num);
      //int sendnumber = UDPclient.send_obj_to_rsu (cur_cluster, cur_cluster_num);
      //cout << "[UDP]:" << sendnumber << endl;

      //log_3D (cur_cluster, cur_cluster_num);
      //log_2D (cur_cluster, cur_cluster_num);

      cout << "[Total running time]:" << timer_algorithm_running.getTimeSeconds () << "s" << endl << endl;

      if (timer_algorithm_running.getTimeSeconds () > 0.25)
      {
        RosModuleA::send_ErrorCode (0x0020);
        cout << "error: low FPS < 4" << endl;
      }

      delete[] cur_cluster;
    }
};

void
detection (int argc,
           char ** argv)
{
  cout.setf (std::ios_base::fixed, std::ios_base::floatfield);
  cout.precision (3);

  //CanModule::initial ();

  RosModuleA::initial (argc, argv);
  RosModuleA::RegisterCallBackLidar (callback_LidFrontTop, callback_LidFrontLeft, callback_LidFrontRight, callback_LidRearLeft, callback_LidRearRight);

  pcl::visualization::PointCloudColorHandlerGenericField<PointXYZI> handler_OD ("intensity");
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

  if (GlobalVariable::ENABLE_DEBUG_MODE)
  {
    pcl::visualization::PointCloudColorHandlerGenericField<PointXYZI> handler_OD ("intensity");
    viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ("PCL HDL Cloud"));
    pcl::visualization::Camera cam;
    cam = CompressFunction ().CamPara (7.122, -7.882, 71.702, 0.991, 0.003, 0.135, 15.755, -0.407, 8.052, 7.489, 143.559, 0.857, 0.857, 0.000, 0.000, 332.000,
                                       1028.000);
    viewer->initCameraParameters ();
    viewer->addCoordinateSystem (3.0);  //red:x green:y
    viewer->setBackgroundColor (0, 0, 0);
    viewer->setCameraParameters (cam, 0);
    viewer->setShowFPS (false);
    viewer->registerKeyboardCallback (KeyboardMouseEvent ().keyboardCallback, (void*) viewer.get ());
    viewer->registerMouseCallback (KeyboardMouseEvent ().mouseCallback, (void*) viewer.get ());
  }

  int viewID = 0;

  ObjectsDetectionA OD;
  OD.initial (viewer, &viewID);

  int ErrorTest[5] = { 0 };

  while (ros::ok ())
  {
    pcl::PointCloud<pcl::PointXYZI>::Ptr release_cloud (new pcl::PointCloud<pcl::PointXYZI>);

    mutex_LidFrontTop.lock ();
    mutex_LidFrontLeft.lock ();
    mutex_LidFrontRight.lock ();
    mutex_LidRearLeft.lock ();
    mutex_LidRearRight.lock ();

    if (cloudPtr_LidFrontTop->is_dense == false && cloudPtr_LidFrontTop->size () > 100)
    {
      *release_cloud += *cloudPtr_LidFrontTop;
      ErrorTest[0] = 0;
    }
    else
    {
      ErrorTest[0]++;
    }

    if (cloudPtr_LidFrontLeft->is_dense == false && cloudPtr_LidFrontLeft->size () > 100)
    {
      *release_cloud += *cloudPtr_LidFrontLeft;
      ErrorTest[1] = 0;
    }
    else
    {
      ErrorTest[1]++;
    }

    if (cloudPtr_LidFrontRight->is_dense == false && cloudPtr_LidFrontRight->size () > 100)
    {
      *release_cloud += *cloudPtr_LidFrontRight;
      ErrorTest[2] = 0;
    }
    else
    {
      ErrorTest[2]++;
    }

    if (cloudPtr_LidRearLeft->is_dense == false && cloudPtr_LidRearLeft->size () > 100)
    {
      *release_cloud += *cloudPtr_LidRearLeft;
      ErrorTest[3] = 0;
    }
    else
    {
      ErrorTest[3]++;
    }

    if (cloudPtr_LidRearRight->is_dense == false && cloudPtr_LidRearRight->size () > 100)
    {
      *release_cloud += *cloudPtr_LidRearRight;
      ErrorTest[4] = 0;
    }
    else
    {
      ErrorTest[4]++;
    }

    cloudPtr_LidFrontTop->is_dense = true;
    cloudPtr_LidFrontLeft->is_dense = true;
    cloudPtr_LidFrontRight->is_dense = true;
    cloudPtr_LidRearLeft->is_dense = true;
    cloudPtr_LidRearRight->is_dense = true;

    GlobalVariable::ROS_TIMESTAMP = max (max (max (max (lidar_time[0], lidar_time[1]), lidar_time[2]), lidar_time[3]), lidar_time[4]);

    mutex_LidFrontTop.unlock ();
    mutex_LidFrontLeft.unlock ();
    mutex_LidFrontRight.unlock ();
    mutex_LidRearLeft.unlock ();
    mutex_LidRearRight.unlock ();

    if (ErrorTest[0] > 100)
    {
      RosModuleA::send_ErrorCode (0x0001);
      cout << "error: LidFrontTop" << endl;
    }

    if (ErrorTest[1] > 100)
    {
      RosModuleA::send_ErrorCode (0x0002);
      cout << "error: LidFrontLeft" << endl;
    }

    if (ErrorTest[2] > 100)
    {
      RosModuleA::send_ErrorCode (0x0003);
      cout << "error: LidFrontRight" << endl;
    }

    if (ErrorTest[3] > 100)
    {
      RosModuleA::send_ErrorCode (0x0004);
      cout << "error: LidRearLeft" << endl;
    }

    if (ErrorTest[4] > 100)
    {
      RosModuleA::send_ErrorCode (0x0005);
      cout << "error: LidRearRight" << endl;
    }

    if (release_cloud->size () > 100)
    {

      OD.setInputCloud (release_cloud);
      OD.update ();
      RosModuleA::send_StitchCloud (release_cloud);

      if (GlobalVariable::ENABLE_DEBUG_MODE)
      {
        handler_OD.setInputCloud (release_cloud);
        if (!viewer->updatePointCloud (release_cloud, handler_OD, "OD"))
        {
          viewer->addPointCloud (release_cloud, handler_OD, "OD");
        }
        viewer->spinOnce ();
      }

      if (GlobalVariable::ENABLE_OUTPUT_PCD)
      {
        if (ErrorTest[0] < 5 && ErrorTest[1] < 5 && ErrorTest[2] < 5 && ErrorTest[3] < 5 && ErrorTest[4] < 5)
        {
          pcl::io::savePCDFileASCII (to_string (GlobalVariable::ROS_TIMESTAMP.sec) + ".pcd", *release_cloud);
        }
      }

    }

    while (get_pause_state ())
      viewer->spinOnce ();

    ros::spinOnce ();
    this_thread::sleep_for (std::chrono::milliseconds (25));

  }
  RosModuleA::send_ErrorCode (0x0006);
  cout << "error: computational loop thread closed" << endl;
}

int
main (int argc,
      char ** argv)
{
  cout << "===================== application startup =====================" << endl;

  GlobalVariable::ENABLE_DEBUG_MODE = pcl::console::find_switch (argc, argv, "-d");
  GlobalVariable::ENABLE_OUTPUT_PCD = pcl::console::find_switch (argc, argv, "-pcd");
  GlobalVariable::ENABLE_LABEL_TOOL = pcl::console::find_switch (argc, argv, "-l");

  thread TheadDetection (detection, argc, argv);

  this_thread::sleep_for (std::chrono::milliseconds (1000));

  if (GlobalVariable::ENABLE_DEBUG_MODE)
  {
    QApplication a (argc, argv);
    QtViewer w;
    w.show ();
    a.exec ();
  }

  while (true)
    this_thread::sleep_for (std::chrono::milliseconds (10000));

  cout << "===================== application end   =====================" << endl;
  return (0);
}

