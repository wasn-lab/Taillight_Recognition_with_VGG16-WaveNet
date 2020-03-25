#include <thread>
#include "glog/logging.h"
#include "pcl_ros/point_cloud.h"

#include "pcd_saver_args_parser.h"
#include "pcd_saver_node.h"
#include "pcd_saver_node_impl.h"

PCDSaverNodeImpl::PCDSaverNodeImpl() = default;
PCDSaverNodeImpl::~PCDSaverNodeImpl() = default;

void PCDSaverNodeImpl::pcd_callback(const sensor_msgs::PointCloud2ConstPtr& in_pcd_message)
{
  auto stamp = in_pcd_message->header.stamp;
  std::thread t(&PCDSaverNodeImpl::save, this, in_pcd_message, stamp.sec, stamp.nsec);
  t.detach();
}

void PCDSaverNodeImpl::save(const sensor_msgs::PointCloud2ConstPtr& in_pcd_message, int sec, int nsec)
{
  char buff[32] = { 0 };
  snprintf(buff, sizeof(buff), "%10d%09d.pcd", sec, nsec);  // NOLINT
  std::string fname(static_cast<const char*>(buff));

  pcl::PointCloud<pcl::PointXYZI>::Ptr pcd_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*in_pcd_message, *pcd_ptr);
  LOG(INFO) << "write " << fname << " points: " << pcd_ptr->points.size();

  pcl::io::savePCDFileASCII(fname, *pcd_ptr);
}

void PCDSaverNodeImpl::subscribe()
{
  std::string topic = pcd_saver::get_pcd_topic();
  subscriber_ = node_handle_.subscribe(topic, 2, &PCDSaverNodeImpl::pcd_callback, this);
}

void PCDSaverNodeImpl::run()
{
  subscribe();
  ros::AsyncSpinner spinner(1);  // number of threads: 1
  spinner.start();
  ros::Rate r(30);  // expected FPS
  while (ros::ok())
  {
    r.sleep();
  }
  spinner.stop();
}
