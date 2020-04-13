#ifndef KEYBOARDMOUSEEVENT_H_
#define KEYBOARDMOUSEEVENT_H_

#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
using namespace pcl;

class KeyboardMouseEvent
{
public:
  KeyboardMouseEvent();
  ~KeyboardMouseEvent();

  static void setCloudToPCD(PointCloud<PointXYZ> input);

  static bool getPauseState();

  static bool getResultState();

  static bool getCHState();

  static bool getBBoxState();

  static bool getLidarAllState();

  static void mouseCallback(const visualization::MouseEvent& event, void* viewer_void);
  static void keyboardCallback(const visualization::KeyboardEvent& event, void* viewer_void);

private:
  static PointCloud<PointXYZ> cloud_save_to_pcd;
  static bool pause_state;
  static bool result_mode_state;
  static bool ch_mode_state;
  static bool bbox_mode_state;
  static bool lidar_all_state;
};

#endif
