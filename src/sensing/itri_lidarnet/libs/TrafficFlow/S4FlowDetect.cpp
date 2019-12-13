#include "S4FlowDetect.h"

S4FlowDetect::S4FlowDetect () :
    viewer (NULL),
    viewID (NULL)
{
}

S4FlowDetect::S4FlowDetect (boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer,
                            int *input_viewID) :
    viewer (input_viewer),
    viewID (input_viewID)
{
}

S4FlowDetect::~S4FlowDetect ()
{
}
void
S4FlowDetect::add_gateway_rectangle (pcl::PointXYZ rectangle_min,
                                     pcl::PointXYZ rectangle_max)
{
  pt_min = rectangle_min;
  pt_max = rectangle_max;
}

void
S4FlowDetect::update (bool is_debug,
                      CLUSTER_INFO* cluster_info,
                      int cluster_number,
                      int *counter)
{
  bool result_has_detected = false;
  static std::vector<CLUSTER_INFO> pre_vehicle_table;

  for (int i = 0; i < cluster_number; i++)
  {
    bool result_point_into = false;

    for (size_t j = 0; j < cluster_info[i].cloud.size (); j++)
    {
      pcl::PointXYZ pt (cluster_info[i].cloud.points[j].x, cluster_info[i].cloud.points[j].y, cluster_info[i].cloud.points[j].z);
      result_point_into |= ! (pt.x < pt_min.x || pt.y < pt_min.y || pt.z < pt_min.z || pt.x > pt_max.x || pt.y > pt_max.y || pt.z > pt_max.z);
      if (result_point_into)
        break;
    }

    if (result_point_into && pcl::geometry::distance (pcl::PointXYZ (0, 0, 0), cluster_info[i].velocity) > 1.5)
    {
      bool flag = true;

      for (std::vector<CLUSTER_INFO>::const_iterator it = pre_vehicle_table.begin (); it != pre_vehicle_table.end ();)
      {
        if (pcl::geometry::distance (cluster_info[i].track_last_center, it->center) < 0.1 && cluster_info[i].track_last_center.x < 99999)
        {
          it = pre_vehicle_table.erase (it);
          flag = false;
          break;
        }
        else if (cluster_info[i].track_last_center.x == 99999)
        {
          flag = false;
          break;
        }
        else
        {
          ++it;
        }
      }

      pre_vehicle_table.push_back (cluster_info[i]);

      if (flag)
      {
        if (cluster_info[i].cluster_tag == 2)
          counter[0]++;
        if (cluster_info[i].cluster_tag == 3)
          counter[1]++;
        if (cluster_info[i].cluster_tag == 4)
          counter[2]++;
      }
      result_has_detected = true;
    }
  }

  if (!result_has_detected)
  {
    pre_vehicle_table.clear ();
  }

  int rgb[3];

  if (result_has_detected)
  {
    rgb[0] = 0;
    rgb[1] = 255;
    rgb[2] = 234;
  }
  else
  {
    rgb[0] = 255;
    rgb[1] = 0;
    rgb[2] = 234;
  }
/*
//   | pt6 _______pt7(max)
     |     |\      \
//   |     | \      \
//   |     |  \______\
//   | pt5 \  |pt2   |pt3
//   |      \ |      |
//   |       \|______|
//   | pt1(min)    pt4
*/

  pcl::PointXYZ pt1 (pt_min.x, pt_min.y, pt_min.z);
  pcl::PointXYZ pt2 (pt_min.x, pt_min.y, pt_max.z);
  pcl::PointXYZ pt3 (pt_max.x, pt_min.y, pt_max.z);
  pcl::PointXYZ pt4 (pt_max.x, pt_min.y, pt_min.z);
  pcl::PointXYZ pt5 (pt_min.x, pt_max.y, pt_min.z);
  pcl::PointXYZ pt6 (pt_min.x, pt_max.y, pt_max.z);
  pcl::PointXYZ pt7 (pt_max.x, pt_max.y, pt_max.z);
  pcl::PointXYZ pt8 (pt_max.x, pt_max.y, pt_min.z);

  viewer->addLine (pt1, pt2, rgb[0], rgb[1], rgb[2], std::to_string (*viewID));
  ++*viewID;
  viewer->addLine (pt1, pt4, rgb[0], rgb[1], rgb[2], std::to_string (*viewID));
  ++*viewID;
  viewer->addLine (pt1, pt5, rgb[0], rgb[1], rgb[2], std::to_string (*viewID));
  ++*viewID;
  viewer->addLine (pt5, pt6, rgb[0], rgb[1], rgb[2], std::to_string (*viewID));
  ++*viewID;
  viewer->addLine (pt5, pt8, rgb[0], rgb[1], rgb[2], std::to_string (*viewID));
  ++*viewID;
  viewer->addLine (pt2, pt6, rgb[0], rgb[1], rgb[2], std::to_string (*viewID));
  ++*viewID;
  viewer->addLine (pt6, pt7, rgb[0], rgb[1], rgb[2], std::to_string (*viewID));
  ++*viewID;
  viewer->addLine (pt7, pt8, rgb[0], rgb[1], rgb[2], std::to_string (*viewID));
  ++*viewID;
  viewer->addLine (pt2, pt3, rgb[0], rgb[1], rgb[2], std::to_string (*viewID));
  ++*viewID;
  viewer->addLine (pt4, pt8, rgb[0], rgb[1], rgb[2], std::to_string (*viewID));
  ++*viewID;
  viewer->addLine (pt3, pt4, rgb[0], rgb[1], rgb[2], std::to_string (*viewID));
  ++*viewID;
  viewer->addLine (pt3, pt7, rgb[0], rgb[1], rgb[2], std::to_string (*viewID));
  ++*viewID;

}

void
S4FlowDetect::show_info (int counter[],
                         double system_running_time_second,
                         double frames_running_time_second)
{
  pcl::visualization::Camera camera;
  viewer->getCameraParameters (camera);
  int interval_y = camera.window_size[1];

  double flow = (double) (counter[0] + counter[1] + counter[2]) / system_running_time_second;
  interval_y = interval_y - 25;
  viewer->addText ("Flow (quantities/time):" + std::to_string (flow), 0, interval_y, 20, 1, 1, 1,  std::to_string (*viewID), 0);
  ++*viewID;

  double Headway = 1 / flow;
  interval_y = interval_y - 25;
  viewer->addText ("Headway (time/quantities):" +  std::to_string (Headway), 0, interval_y, 20, 1, 1, 1,  std::to_string (*viewID), 0);
  ++*viewID;

  interval_y = interval_y - 25;
  viewer->addText ("Counter", 0, interval_y, 20, 1, 1, 1,  std::to_string (*viewID), 0);
  ++*viewID;

  interval_y = interval_y - 25;
  viewer->addText ("Motorcycle:" + std::to_string (counter[0]), 0, interval_y, 20, 1, 1, 1, std::to_string (*viewID), 0);
  ++*viewID;

  interval_y = interval_y - 25;
  viewer->addText ("Car:" + std::to_string (counter[1]), 0, interval_y, 20, 1, 1, 1, std::to_string (*viewID), 0);
  ++*viewID;

  interval_y = interval_y - 25;
  viewer->addText ("Bus:" + std::to_string (counter[2]), 0, interval_y, 20, 1, 1, 1, std::to_string (*viewID), 0);
  ++*viewID;

}

/*
 bool
 gateway_detection (boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
 int *viewID,
 int cluster_number,
 CLUSTER_INFO* cluster_info,
 PointXYZ pt_min,
 PointXYZ pt_max,
 int *counter)
 {
 bool result_has_detected = false;
 static vector<CLUSTER_INFO> pre_detect_table;

 for (int i = 0; i < cluster_number; i++)
 {
 bool result_point_into = false;

 for (int j = 0; j < cluster_info[i].cloud.size (); j++)
 {
 PointXYZ pt (cluster_info[i].cloud.points[j].x, cluster_info[i].cloud.points[j].y, cluster_info[i].cloud.points[j].z);
 result_point_into |= ! (pt.x < pt_min.x || pt.y < pt_min.y || pt.z < pt_min.z || pt.x > pt_max.x || pt.y > pt_max.y || pt.z > pt_max.z);
 if(result_point_into)break;
 }

 if (result_point_into)
 {
 bool flag = true;

 float lock_num = 99999, lock_dis = 99999, lock_vol = 99999;
 int index_num = -1, index_dis = -1, index_vol = -1;

 for (int k = 0; k < pre_detect_table.size (); k++)
 {
 float num = abs (cluster_info[i].cloud.size () - pre_detect_table.at (k).cloud.size ());
 float dis = geometry::distance (cluster_info[i].center, pre_detect_table.at (k).center);
 float vol = fabs (cluster_info[i].dis_max_min - pre_detect_table.at (k).dis_max_min);

 float cos_theta = 0;
 if (cluster_info[i].found_num == true && pre_detect_table.at (k).found_num == true)
 {
 float cur_spd = sqrt (
 cluster_info[i].velocity.x * cluster_info[i].velocity.x + cluster_info[i].velocity.y * cluster_info[i].velocity.y
 + cluster_info[i].velocity.z * cluster_info[i].velocity.z);
 cos_theta = (cluster_info[i].velocity.x * pre_detect_table.at (k).velocity.x + cluster_info[i].velocity.y * pre_detect_table.at (k).velocity.y
 + cluster_info[i].velocity.z * pre_detect_table.at (k).velocity.z) / (dis * cur_spd);
 }

 if (dis < 5)
 {
 if (num < lock_num && cluster_info[i].cluster_tag == pre_detect_table.at (k).cluster_tag)
 {
 lock_num = num;
 index_num = k;
 }
 if (dis < lock_dis && cluster_info[i].cluster_tag == pre_detect_table.at (k).cluster_tag)
 {
 lock_dis = dis;
 index_dis = k;
 }
 if (vol < lock_vol && cluster_info[i].cluster_tag == pre_detect_table.at (k).cluster_tag)
 {
 lock_vol = vol;
 index_vol = k;
 }
 }
 }

 //cout<<"------------------------------------"<<lock_dis<<"---------"<<lock_vol<<"---------"<<lock_num<<"---------"<<pre_spd<<endl;
 if (lock_dis < 3)
 {
 pre_detect_table.erase (pre_detect_table.begin () + index_dis);
 flag = false;
 //cout << "-----------------------------------------------------------1" << endl;
 }
 else if (lock_dis < 5)
 {
 if (lock_vol < 0.3)
 {
 pre_detect_table.erase (pre_detect_table.begin () + index_vol);
 flag = false;
 //cout << "-----------------------------------------------------------2" << endl;
 }
 else if (lock_num <= 100)
 {
 pre_detect_table.erase (pre_detect_table.begin () + index_num);
 flag = false;
 //cout << "-----------------------------------------------------------3" << endl;
 }
 }

 pre_detect_table.push_back (cluster_info[i]);

 if (flag)
 {
 if (cluster_info[i].cluster_tag == 2)
 counter[0]++;
 if (cluster_info[i].cluster_tag == 3)
 counter[1]++;
 if (cluster_info[i].cluster_tag == 4)
 counter[2]++;
 }

 result_has_detected = true;
 }

 }

 if (!result_has_detected)
 {
 pre_detect_table.clear ();
 }

 return result_has_detected;
 }
 */

