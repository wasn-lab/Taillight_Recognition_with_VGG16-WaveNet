#include "ros_gmphd.h"
#include "stdafx.h"
#include <unistd.h>
#include "gmphd_def.h"
#include <sstream>
#include <string.h>

namespace ped
{
void ROSGMPHD::run()
{
  pedestrian_event();
}

void ROSGMPHD::chatter_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  count++;
  ros::Time start, stop;
  start = ros::Time::now();

  vector<vector<float>> seq_dets_front;
  vector<vector<float>> seq_dets_fov30;
  vector<vector<float>> seq_dets_left_back;
  vector<vector<float>> seq_dets_right_back;

  for (auto const& obj : msg->objects)
  {
    /************* Make msg to GMPHD input ***************/
    vector<float> tmp_det;
    tmp_det.push_back((float)count);
    tmp_det.push_back((float)obj.camInfo[0].u);
    tmp_det.push_back((float)obj.camInfo[0].v);
    tmp_det.push_back((float)obj.camInfo[0].width);
    tmp_det.push_back((float)obj.camInfo[0].height);
    tmp_det.push_back((float)obj.camInfo[0].prob);
    if (obj.camInfo[0].id == 0)  // front
    {
      seq_dets_front.push_back(tmp_det);
    }
    if (obj.camInfo[0].id == 1)  // fov30
    {
      seq_dets_fov30.push_back(tmp_det);
    }
    if (obj.camInfo[0].id == 9)  // left back
    {
      seq_dets_left_back.push_back(tmp_det);
    }
    if (obj.camInfo[0].id == 6)  // right back
    {
      seq_dets_right_back.push_back(tmp_det);
    }
  }
  vector<float> tmp_det;
  tmp_det.push_back((float)count);
  tmp_det.push_back((float)(-1));
  tmp_det.push_back((float)(-1));
  tmp_det.push_back((float)(-1));
  tmp_det.push_back((float)(-1));
  tmp_det.push_back((float)(-1));
  if (seq_dets_front.empty())
  {
    seq_dets_front.push_back(tmp_det);
  }
  if (seq_dets_fov30.empty())
  {
    seq_dets_fov30.push_back(tmp_det);
  }
  if (seq_dets_left_back.empty())
  {
    seq_dets_left_back.push_back(tmp_det);
  }
  if (seq_dets_right_back.empty())
  {
    seq_dets_right_back.push_back(tmp_det);
  }

  /*   test   */
  if (count == 2 || true)
  {
    for (unsigned int _k = 0; _k < seq_dets_front.size(); _k++)
    {
      for (unsigned int _k2 = 0; _k2 < seq_dets_front[0].size(); _k2++)
      {
        std::cout << seq_dets_front[_k][_k2] << ",";
      }
      std::cout << std::endl;
    }
    std::cout << "\nend" << std::endl;
  }
  /*   test   */

  /***** DoMOT *******/
  tracker_front->SetTotalFrames(count);
  vector<vector<float>> tracks_front = tracker_front->DoMOT(count - 1, seq_dets_front);
  tracker_fov30->SetTotalFrames(count);
  vector<vector<float>> tracks_fov30 = tracker_fov30->DoMOT(count - 1, seq_dets_fov30);
  tracker_left_back->SetTotalFrames(count);
  vector<vector<float>> tracks_left_back = tracker_left_back->DoMOT(count - 1, seq_dets_left_back);
  tracker_right_back->SetTotalFrames(count);
  vector<vector<float>> tracks_right_back = tracker_right_back->DoMOT(count - 1, seq_dets_right_back);

  /*   test   */
  if (count == 2 || true)
  {
    for (unsigned int _k = 0; _k < tracks_front.size(); _k++)
    {
      for (unsigned int _k2 = 0; _k2 < tracks_front[0].size(); _k2++)
      {
        std::cout << tracks_front[_k][_k2] << ",";
      }
      std::cout << std::endl;
    }
  }
  /*   test   */

  msgs::DetectedObjectArray pub_array_front;
  pub_array_front.header = msg->header;
  msgs::DetectedObjectArray pub_array_fov30;
  pub_array_fov30.header = msg->header;
  msgs::DetectedObjectArray pub_array_left_back;
  pub_array_left_back.header = msg->header;
  msgs::DetectedObjectArray pub_array_right_back;
  pub_array_right_back.header = msg->header;

  for (auto obj : msg->objects)
  {
    /******** compare & tracks to msg *********/
    for (unsigned int i = 0; i < tracks_front.size(); i++)
    {
      if (obj.camInfo[0].u == tracks_front[i][1] && obj.camInfo[0].v == tracks_front[i][2] &&
          obj.camInfo[0].width == tracks_front[i][3] && obj.camInfo[0].height == tracks_front[i][4])
      {
        obj.track.id = tracks_front[i][0];
        pub_array_front.objects.emplace_back(obj);
      }
    }
    for (unsigned int i = 0; i < tracks_fov30.size(); i++)
    {
      if (obj.camInfo[0].u == tracks_fov30[i][1] && obj.camInfo[0].v == tracks_fov30[i][2] &&
          obj.camInfo[0].width == tracks_fov30[i][3] && obj.camInfo[0].height == tracks_fov30[i][4])
      {
        obj.track.id = tracks_fov30[i][0];
        pub_array_fov30.objects.emplace_back(obj);
      }
    }
    for (unsigned int i = 0; i < tracks_left_back.size(); i++)
    {
      if (obj.camInfo[0].u == tracks_left_back[i][1] && obj.camInfo[0].v == tracks_left_back[i][2] &&
          obj.camInfo[0].width == tracks_left_back[i][3] && obj.camInfo[0].height == tracks_left_back[i][4])
      {
        obj.track.id = tracks_left_back[i][0];
        pub_array_left_back.objects.emplace_back(obj);
      }
    }
    for (unsigned int i = 0; i < tracks_right_back.size(); i++)
    {
      if (obj.camInfo[0].u == tracks_right_back[i][1] && obj.camInfo[0].v == tracks_right_back[i][2] &&
          obj.camInfo[0].width == tracks_right_back[i][3] && obj.camInfo[0].height == tracks_right_back[i][4])
      {
        obj.track.id = tracks_right_back[i][0];
        pub_array_right_back.objects.emplace_back(obj);
      }
    }
  }

  /********* publish ************/
  chatter_pub_front.publish(pub_array_front);
  chatter_pub_fov30.publish(pub_array_fov30);
  chatter_pub_left_back.publish(pub_array_left_back);
  chatter_pub_right_back.publish(pub_array_right_back);

  stop = ros::Time::now();
  total_time += stop - start;

#if PRINT_MESSAGE
  std::cout << "Cost time: " << stop - start << " sec" << std::endl;
  std::cout << "Total time: " << total_time << " sec / loop: " << count << std::endl;
#endif
}

void ROSGMPHD::pedestrian_event()
{
  // AsyncSpinner reference:
  //  https://gist.github.com/bgromov/45ebeced9e8067d9f13cceececc00d5b#file-test_spinner-cpp-L63

  // This node handle uses global callback queue
  ros::NodeHandle nh_sub_1;

  // Set custom callback queue

  ros::Subscriber sub_1;  // nh_sub_1

  sub_1 = nh_sub_1.subscribe("/CameraDetection", 1, &ROSGMPHD::chatter_callback,
                             this);  // /PathPredictionOutput is sub topic

  // Loop with 100 Hz rate
  ros::Rate loop_rate(20);
  while (ros::ok())
  {
    // Process messages on global callback queue
    ros::spinOnce();
    loop_rate.sleep();
  }
  // Wait for ROS threads to terminate
  ros::waitForShutdown();
}

}  // namespace ped
int main(int argc, char** argv)
{
  ros::Time::init();
  ros::Time start, stop;
  start = ros::Time::now();
  ros::init(argc, argv, "ros_gmphd");

  ped::ROSGMPHD ros_gmphd;
  /************  Initial GMPHD param *****************/
  GMPHDOGMparams sceneParam;
  // setting param
  sceneParam.DET_MIN_CONF = -100;     // Detection Confidence Threshold
  sceneParam.T2TA_MAX_INTERVAL = 80;  // T2TA Maximum Interval
  sceneParam.TRACK_MIN_SIZE = 2;      // Track Minium Length
  sceneParam.FRAMES_DELAY_SIZE = sceneParam.TRACK_MIN_SIZE - 1;
  sceneParam.GROUP_QUEUE_SIZE = sceneParam.TRACK_MIN_SIZE * 10;
  //
  ros_gmphd.tracker_front->SetParams(sceneParam);
  ros_gmphd.tracker_fov30->SetParams(sceneParam);
  ros_gmphd.tracker_left_back->SetParams(sceneParam);
  ros_gmphd.tracker_right_back->SetParams(sceneParam);

  ros::NodeHandle nh1;
  ros_gmphd.chatter_pub_front = nh1.advertise<msgs::DetectedObjectArray>("/Tracking2D/front_bottom_60", 1);
  ros::NodeHandle nh2;
  ros_gmphd.chatter_pub_fov30 = nh2.advertise<msgs::DetectedObjectArray>("/Tracking2D/front_top_far_30", 1);
  ros::NodeHandle nh3;
  ros_gmphd.chatter_pub_left_back = nh3.advertise<msgs::DetectedObjectArray>("/Tracking2D/left_back_60", 1);
  ros::NodeHandle nh4;
  ros_gmphd.chatter_pub_right_back = nh4.advertise<msgs::DetectedObjectArray>("/Tracking2D/right_back_60", 1);
  stop = ros::Time::now();
  std::cout << "GMPHD started. Init time: " << stop - start << " sec" << std::endl;

  ros_gmphd.run();
  return 0;
}
