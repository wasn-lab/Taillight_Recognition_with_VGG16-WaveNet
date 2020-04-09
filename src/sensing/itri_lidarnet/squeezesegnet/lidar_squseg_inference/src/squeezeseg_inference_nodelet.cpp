#include "squeezeseg_inference_nodelet.h"


ros::Publisher ssn_nodelet::nn_pub;
ros::Subscriber ssn_nodelet::LidarAllSub;

string ssn_nodelet::data_set;
char ssn_nodelet::ViewType;
int ssn_nodelet::pub_type;
bool ssn_nodelet::hybrid_detect;
bool ssn_nodelet::debug_output;

string ssn_nodelet::GET_data_set;
string ssn_nodelet::GET_ViewType;
string ssn_nodelet::GET_pub_type;
string ssn_nodelet::GET_hybrid_detect;

vector<TF_inference> ssn_nodelet::SSN_all;


void
ssn_nodelet::LidarsNodelet::onInit()
{
  ros::NodeHandle nh;

  nh.getParam("SSN_DataSet", GET_data_set);
  nh.getParam("SSN_ViewType", GET_ViewType);
  nh.getParam("SSN_PubType", GET_pub_type);
  nh.getParam("SSN_Hybrid_Detect", GET_hybrid_detect);

  // check debug mode
  ros::param::get("/debug_output", debug_output);

  data_set = GET_data_set;
  ViewType = GET_ViewType.at(0);
  pub_type = stoi(GET_pub_type);
  hybrid_detect = to_bool(GET_hybrid_detect);

  cout << "#=========== SSN NODELET INIT" << endl;
  cout << "data_set: " << data_set << endl;
  cout << "ViewType: " << ViewType << endl;
  cout << "pub_type: " << pub_type << endl;
  cout << "hybird_detect: " << hybrid_detect << endl;
  cout << "debug_output: " << debug_output << endl;  

  LidarAllSub = nh.subscribe("/LidarAll/NonGround", 1, callback_LidarAll);
  nn_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZIL>>("/squ_seg/result_cloud", 1);

  // =============== Tensorflow ==================
  vector<float> phi_center_all = phi_center_grid(ViewType);

  for (size_t i = 0; i < phi_center_all.size(); i++)
    SSN_all.push_back(TF_inference(data_set, ViewType, phi_center_all.at(i), pub_type));

  vector<int> TF_ERcode(phi_center_all.size());
  for (size_t i = 0; i < phi_center_all.size(); i++)
    TF_ERcode.at(i) = SSN_all.at(i).TF_init();

  // TODO
  /*
   for(size_t i=0; i < phi_center_all.size(); i++)
   SSN_all.at(i).TF_quit();
   */
}

void 
ssn_nodelet::LidarsNodelet::callback_LidarAll(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& msg)
{
  // cout << "TensorFlow Version: " << TF_Version() << endl;
  
  if (debug_output)
  { 
    ros::Time rosTime;
    pcl_conversions::fromPCL (msg->header.stamp, rosTime);
    cout << "[Top->SSN]: " << (ros::Time::now () - rosTime).toSec() *1000 << "ms"  << endl;
  }
  
  pcl::StopWatch stopWatch;

  VPointCloud::Ptr release_Cloud(new VPointCloud);

  *release_Cloud = *msg;

  if (release_Cloud->size() > 100)
  {
    //   ===============  temporally enable part of rule-based code for ensuring front-view detection
    //   ==========================
    VPointCloud::Ptr select_Cloud(new VPointCloud);
    if (hybrid_detect == 1)
    {
      *select_Cloud = CuboidFilter().pass_through_soild<PointXYZI>(release_Cloud, -30, 50, -3.0, 3.0, -3.12, 0);
    }
    //   ========================================================================================================================

    // the origin of coordinate is shifted -2 m in x-axis and -1.4 m in z-axis in SSN to maximize projection ratio
    // Note that following shift will be recovered in TF_inference::TF_run() !!!!!!!!!!!!!1
    const float x_projCenter = proj_center(data_set, 0);
    const float z_projCenter = proj_center(data_set, 1);

    for (size_t i = 0; i < release_Cloud->points.size(); i++)
    {
      release_Cloud->points[i].x = release_Cloud->points[i].x - x_projCenter;
      // float release_Cloud->points[i].y = release_Cloud->points[i].y;
      release_Cloud->points[i].z = release_Cloud->points[i].z - z_projCenter;
    }

    vector<VPointCloudXYZIL::Ptr> result_cloud;
    for (size_t i = 0; i < SSN_all.size(); i++)
      result_cloud.push_back(VPointCloudXYZIL::Ptr(new VPointCloudXYZIL));

      // VPointCloudXYZIL::Ptr result_cloud(new VPointCloudXYZIL);

#if (false)
    vector<thread> mthreads;
    for (size_t i = 0; i < SSN_all.size(); i++)
    {
      mthreads.push_back(thread(&TF_inference::TF_run, &SSN_all.at(i), release_Cloud, result_cloud.at(i)));
      // ros::Duration(0.02).sleep();
    }

    for (size_t i = 0; i < SSN_all.size(); i++)
      mthreads.at(i).join();
#else
    SSN_all.at(0).TF_run(release_Cloud, result_cloud.at(0));
    SSN_all.at(1).TF_run(release_Cloud, result_cloud.at(1));
    SSN_all.at(2).TF_run(release_Cloud, result_cloud.at(2));
    SSN_all.at(3).TF_run(release_Cloud, result_cloud.at(3));
#endif

    VPointCloudXYZIL::Ptr result_cloud_all(new VPointCloudXYZIL);

    for (size_t i = 0; i < SSN_all.size(); i++)
      *result_cloud_all += *result_cloud.at(i);

    result_cloud_all->header.frame_id = msg->header.frame_id;
    result_cloud_all->header.stamp = msg->header.stamp;
    // pcl_conversions::toPCL(ros::Time::now(), result_cloud.header.stamp);
    // result_cloud.header.seq = msg->header.seq;

    if (hybrid_detect)
    {
      for (size_t i = 0; i < (size_t)select_Cloud->points.size(); i++)
      {
        VPointXYZIL pointinfo;
        pointinfo.x = select_Cloud->points.at(i).x;
        pointinfo.y = select_Cloud->points.at(i).y;
        pointinfo.z = select_Cloud->points.at(i).z;
        pointinfo.intensity = select_Cloud->points.at(i).intensity;
        pointinfo.label = 4;

        result_cloud_all->push_back(pointinfo);
      }
    }

    nn_pub.publish(*result_cloud_all);
    result_cloud_all->clear();

    // ======== following comment used for debugging of subscription ========
    // sensor_msgs::PointCloud2 all_msg;
    // pcl::toROSMsg (*release_Cloud, all_msg);
    // all_msg.header.frame_id = "lidar";
    // all_msg.header.stamp = ros::Time::now ();
    // all_msg.header.seq = msg->header.seq;
    // all_pub.publish (all_msg);  // publish to /release_cloud
  }

  if (debug_output)
  {
      cout << "[SSN]: " << stopWatch.getTimeSeconds() << "s" << endl;
  } 
}

bool
ssn_nodelet::LidarsNodelet::to_bool(std::string const& s)
{
  return s != "0";
}

PLUGINLIB_EXPORT_CLASS(ssn_nodelet::LidarsNodelet, nodelet::Nodelet)
