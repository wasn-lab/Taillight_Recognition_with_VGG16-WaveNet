#include <thread>
#include "glog/logging.h"
#include "pcl_ros/point_cloud.h"

#include "pcd_to_pc2_args_parser.h"
#include "pcd_to_pc2_node.h"

namespace pcd_to_pc2
{
PCDToPc2Node::PCDToPc2Node()
{
}

PCDToPc2Node::~PCDToPc2Node() = default;

void PCDToPc2Node::load_pcd()
{
  auto pcd_path = get_pcd_path();
  CHECK(!pcd_path.empty()) << "Please specify a valid pcd file path";

  auto ret = pcl::io::loadPCDFile(pcd_path, pc2_);
  CHECK(ret >= 0) << "Fail to load pcd file " << pcd_path;
  LOG(INFO) << "Successfully load " << pcd_path;

  pc2_.header.frame_id = get_frame_id();

  std::string field_names;

  LOG(INFO) << " number of points: " << pc2_.width * pc2_.height;
  for(auto& field: pc2_.fields)
  {
    field_names += field.name + " ";
  }
  field_names.pop_back();
  LOG(INFO) << " fields: " << pc2_.fields.size() << " (" << field_names << ")";
  LOG(INFO) << " frame_id: " << pc2_.header.frame_id;
}

void PCDToPc2Node::run()
{
  load_pcd();
  auto pub = node_handle_.advertise<sensor_msgs::PointCloud2>(get_output_topic(), 1);

  ros::Rate r(get_fps());
  LOG(INFO) << "Publish pcd at " << get_output_topic() << ", FPS = " << get_fps();

  while (ros::ok())
  {
    pc2_.header.stamp = ros::Time::now();
    pub.publish(pc2_);
    r.sleep();
  }
}
};  // namespace pcd_to_pc2
