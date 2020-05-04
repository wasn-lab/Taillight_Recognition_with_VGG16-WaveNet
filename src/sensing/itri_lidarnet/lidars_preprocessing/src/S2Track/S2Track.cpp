#include "S2Track.h"

S2Track::S2Track() : viewer(NULL), viewID(NULL)
{
  counter_direction = 0;
}

S2Track::S2Track(boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer, int* input_viewID)
  : viewer(input_viewer), viewID(input_viewID)
{
  counter_direction = 0;
}

S2Track::~S2Track()
{
}

void S2Track::update(bool is_debug, CLUSTER_INFO* cluster_info, int cluster_size)
{
  //  keep in mind
  //  pre(i)   cur(j)
  //     a --- 1
  //     b -/  2
  //     c /   3
  for (int i = 0; i < cluster_size; i++)  // i = cur
  {
    float lock_dis = 99999, lock_vol = 99999, lock_num = 99999, lock_factor = 99999, lock_factor2 = 99999;
    int index_dis = -1, index_vol = -1, index_num = -1, index_factor = -1, index_factor2 = -1;
    // float cos_theta = 0;

    // start scanning all previous vehicle
    for (size_t j = 0; j < pre_vehicle_table.size(); j++)  // j = pre
    {
      float dif_dis = geometry::distance(PointXYZ(cluster_info[i].center.x, cluster_info[i].center.y, 0),
                                         PointXYZ(pre_vehicle_table.at(j).center.x, pre_vehicle_table.at(j).center.y,
                                                  0));  // the difference of cur center and pre center
      float dif_vol = fabs(cluster_info[i].dis_max_min - pre_vehicle_table.at(j).dis_max_min);
      float dif_num = abs(int(cluster_info[i].cloud.size()) - int(pre_vehicle_table.at(j).cloud.size()));

      PointXYZ test_direction(cluster_info[i].center.x - pre_vehicle_table.at(j).center.x,
                              cluster_info[i].center.y - pre_vehicle_table.at(j).center.y, 0);

      if (dif_dis < lock_dis && (test_direction.x != 0 || test_direction.y != 0))
      {
        lock_dis = dif_dis;
        index_dis = j;
      }
      else if (dif_dis < 0.3)
      {
        lock_dis = dif_dis;
        index_dis = j;
      }

      if (pre_vehicle_table.at(j).found_num > 0)  // if it has pre data
      {
        float pre_sx = pre_vehicle_table.at(j).velocity.x;
        float pre_sy = pre_vehicle_table.at(j).velocity.y;
        float cur_sx = (cluster_info[i].center.x - pre_vehicle_table.at(j).center.x) / frame_time.getTimeSeconds();
        float cur_sy = (cluster_info[i].center.y - pre_vehicle_table.at(j).center.y) / frame_time.getTimeSeconds();
        float cur_spd = sqrt(cur_sx * cur_sx + cur_sy * cur_sy);
        float dif_spd = fabs(cur_spd - geometry::distance(PointXYZ(pre_vehicle_table.at(j).velocity.x,
                                                                   pre_vehicle_table.at(j).velocity.y, 0),
                                                          PointXYZ(0, 0, 0)));
        float cos_theta = (cur_sx * pre_sx + cur_sy * pre_sy) / (dif_dis * cur_spd);  // vector inner product

        if (dif_dis > 0.5 && dif_dis < 10 && dif_vol < 3 && dif_num < 100 && dif_spd < 30 && fabs(cos_theta) < 0.5)
        {
          if (dif_vol < lock_vol)
          {
            lock_vol = dif_vol;
            index_vol = j;
          }
          if (dif_num < lock_num)
          {
            lock_num = dif_num;
            index_num = j;
          }

          float dif_pdt = geometry::distance(
              cluster_info[i].center,
              pre_vehicle_table.at(j).predict_next_center);  // the difference of cur center and pre predict_next_center
          float dif_factor = dif_pdt + dif_vol + dif_dis;

          if (dif_factor < lock_factor)
          {
            lock_factor = dif_factor;
            index_factor = j;
          }
        }
      }
      else if (dif_dis < 10 && dif_vol < 3 && dif_num < 300)  // if it doesn't have pre data
      {
        if (dif_vol < lock_vol)
        {
          lock_vol = dif_vol;
          index_vol = j;
        }
        if (dif_num < lock_num)
        {
          lock_num = dif_num;
          index_num = j;
        }

        float dif_factor2 = dif_vol + dif_dis + dif_num;

        if (dif_factor2 < lock_factor)
        {
          lock_factor2 = dif_factor2;
          index_factor2 = j;
        }
      }
    }

    // scan completed! now we decide who is the most similar object
    int the_best_index = -1;

    if (lock_dis < 1.8)
    {
      the_best_index = index_dis;
      // if (is_debug) viewer->addText3D ("F", cluster_info[i].center, 1, 1, 1, 1, to_string (*viewID), 0);
    }
    else if (lock_dis < 2.0 && index_vol == index_num && index_num == index_dis)
    {
      the_best_index = index_dis;
      if (is_debug)
      {
        viewer->addText3D("C", cluster_info[i].center, 1, 1, 1, 1, to_string(*viewID), 0);
      }
    }
    else if (lock_dis < 2.5 && (index_dis == index_vol))
    {
      the_best_index = index_vol;
      if (is_debug)
      {
        viewer->addText3D("D", cluster_info[i].center, 1, 1, 1, 1, to_string(*viewID), 0);
      }
    }
    else if (lock_dis < 3.0 && (index_dis == index_num))
    {
      the_best_index = index_num;
      if (is_debug)
      {
        viewer->addText3D("E", cluster_info[i].center, 1, 1, 1, 1, to_string(*viewID), 0);
      }
    }
    else if (lock_factor < 3.0)
    {
      the_best_index = index_factor;
      if (is_debug)
      {
        viewer->addText3D("A", cluster_info[i].center, 1, 1, 1, 1, to_string(*viewID), 0);
      }
    }
    else if (lock_factor2 <= 3.0)
    {
      the_best_index = index_factor2;
      if (is_debug)
      {
        viewer->addText3D("B", cluster_info[i].center, 1, 1, 1, 1, to_string(*viewID), 0);
      }
    }
    else
    {
      if (lock_dis == 99999)
      {
        if (is_debug)
        {
          viewer->addText3D("-", cluster_info[i].center, 1, 1, 1, 1, to_string(*viewID), 0);
        }
        else if (is_debug)
        {
          viewer->addText3D(to_string((int)lock_dis), cluster_info[i].center, 2, 1, 1, 1, to_string(*viewID), 0);
        }
      }
      else
      {
        if (is_debug)
        {
          viewer->addText3D("*", cluster_info[i].center, 1, 1, 1, 1, to_string(*viewID), 0);
        }
      }
    }
    ++*viewID;

    if (the_best_index >= 0)
    {
      if (is_debug)
      {
        viewer->addLine(cluster_info[i].center, pre_vehicle_table.at(the_best_index).center, 0.0, 1.0, 0.0,
                        to_string(*viewID));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, to_string(*viewID));
        ++*viewID;
      }

      if (pre_vehicle_table.at(the_best_index).found_num > 0)
      {
        cluster_info[i].tracking_id = pre_vehicle_table.at(the_best_index).tracking_id;
      }
      else
      {
        ++tracking_id_count;
        cluster_info[i].tracking_id = tracking_id_count;
      }

      cluster_info[i].track_last_center = pre_vehicle_table.at(the_best_index).center;
      cluster_info[i].velocity.x =
          (cluster_info[i].center.x - pre_vehicle_table.at(the_best_index).center.x) / frame_time.getTimeSeconds();
      cluster_info[i].velocity.y =
          (cluster_info[i].center.y - pre_vehicle_table.at(the_best_index).center.y) / frame_time.getTimeSeconds();
      cluster_info[i].velocity.z =
          (cluster_info[i].center.z - pre_vehicle_table.at(the_best_index).center.z) / frame_time.getTimeSeconds();
      cluster_info[i].predict_next_center.x =
          cluster_info[i].center.x + pre_vehicle_table.at(the_best_index).velocity.x;
      cluster_info[i].predict_next_center.y =
          cluster_info[i].center.y + pre_vehicle_table.at(the_best_index).velocity.y;
      cluster_info[i].predict_next_center.z =
          cluster_info[i].center.z + pre_vehicle_table.at(the_best_index).velocity.z;
      cluster_info[i].found_num = pre_vehicle_table.at(the_best_index).found_num + 1;  // represent the object be found

      pre_vehicle_table.erase(pre_vehicle_table.begin() + the_best_index);
    }

    if (tracking_id_count > 256)
    {
      tracking_id_count = 0;
    }

    if (is_debug)
    {
      viewer->addText3D(to_string(cluster_info[i].tracking_id), cluster_info[i].center, 1, 1, 1, 1, to_string(*viewID),
                        0);
      ++*viewID;

      // int speed = geometry::distance (cluster_info[i].velocity, PointXYZ (0, 0, 0)) * 3.6;
      // viewer->addText3D (to_string(speed), cluster_info[i].center, 1, 1, 1, 1, to_string (*viewID), 0);
      ++*viewID;

      // viewer->addText3D (to_string ((int) (cluster_info[i].tracking_id)) + "id", cluster_info[i].max, 0.7, 255, 255,
      // 255, to_string (*viewID));
      ++*viewID;
    }
  }

  pre_vehicle_table.clear();
  pre_vehicle_table.resize(cluster_size);

#pragma omp parallel for
  for (int i = 0; i < cluster_size; i++)  // if the table did not have this cluster , add it
  {
    pre_vehicle_table.at(i) = cluster_info[i];
  }

  frame_time.reset();
}
