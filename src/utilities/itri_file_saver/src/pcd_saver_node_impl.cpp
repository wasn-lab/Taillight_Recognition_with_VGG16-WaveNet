#include <thread>
#include "glog/logging.h"
#include "pc_transform_utils.h"
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
  snprintf(buff, sizeof(buff), "%10d.%09d.pcd", sec, nsec);  // NOLINT
  std::string fname(static_cast<const char*>(buff));

  auto cloud_ptr = pc_transform::pc2_msg_to_xyzi(in_pcd_message);
  pcl::PCDWriter writer;

  auto width = cloud_ptr->width;
  auto height = cloud_ptr->height;
  auto npoints = cloud_ptr->points.size();
  LOG(INFO) << "write " << fname << ", points: " << npoints << ", width: " << width << ", height: " << height
            << ", is_dense: " << static_cast<bool>(in_pcd_message->is_dense)
            << ", point_step: " << in_pcd_message->point_step << ", raw_step: " << in_pcd_message->row_step
            << ", num_fields: " << in_pcd_message->fields.size();

  if (pcd_saver::save_as_ascii()) {
    writer.writeASCII(fname, *cloud_ptr);
  } else {
    // writer.writeBinary(fname, pc2);
    writer.writeBinaryCompressed(fname, *cloud_ptr);
  }

  /*
  for (size_t i = 0; i < in_pcd_message->fields.size(); i++)
  {
    LOG(INFO) << "fields[" << i << "]:"
              << " name: " << in_pcd_message->fields[i].name
              << ", datatype: " << static_cast<int>(in_pcd_message->fields[i].datatype)
              << ", offset: " << in_pcd_message->fields[i].offset << ", count: " << in_pcd_message->fields[i].count;
  }
  */
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
