#include "ros_gmphd.h"
#include "stdafx.h"
#include <unistd.h>
#include "gmphd_def.h"
#include <sstream>
#include <string.h>


namespace ped
{
void BagToMOT::run()
{
  pedestrian_event();
}

void BagToMOT::chatter_callback(const msgs::DetectedObjectArray::ConstPtr& msg)
{
  count++;
  ros::Time start, stop;
  start = ros::Time::now();
  
  vector<vector<float>> seqDets;

  for (auto const& obj : msg->objects)
  {
    // set msg infomation
    msgs::PedObject obj_pub;
    obj_pub.camInfo = obj.camInfo;
    obj_pub.camInfo.u = obj.camInfo.u;
    obj_pub.camInfo.v = obj.camInfo.v;
    obj_pub.camInfo.width = obj.camInfo.width;
    obj_pub.camInfo.height = obj.camInfo.height;
    obj_pub.camInfo.prob = obj.camInfo.prob;

    // Check object source is camera
    if (obj.camInfo.width == 0 || obj_pub.camInfo.height == 0)
    {
      continue;
    }
    /************* Make msg to GMPHD input ***************/
    vector<float> tmp_det;
    tmp_det.push_back((float)count);
    tmp_det.push_back((float)obj_pub.camInfo.u);
    tmp_det.push_back((float)obj_pub.camInfo.v);
    tmp_det.push_back((float)obj_pub.camInfo.width);
    tmp_det.push_back((float)obj_pub.camInfo.height);
    tmp_det.push_back((float)obj_pub.camInfo.prob);
    seqDets.push_back(tmp_det);
  }
  if (seqDets.empty())
  {
    vector<float> tmp_det;
    tmp_det.push_back((float)count);
    tmp_det.push_back((float)-1);
    tmp_det.push_back((float)-1);
    tmp_det.push_back((float)-1);
    tmp_det.push_back((float)-1);
    tmp_det.push_back((float)-1);
    seqDets.push_back(tmp_det);
  }
  
  /*   test   */
  if(count==2 || true)
  {
    for(unsigned int _k = 0;_k<seqDets.size();_k++)
    {
      for(unsigned int _k2=0;_k2<seqDets[0].size();_k2++)
      {
        cout<<seqDets[_k][_k2]<<",";
      }
      cout<<endl;
    }
    cout<<"\nend\n";
  }
  /*   test   */
  
  /***** DoMOT *******/
  tracker->SetTotalFrames(count);
  vector<vector<float>> tracks = tracker->DoMOT(count-1, seqDets);
  
  /*   test   */
  if(count==2  || true)
  {
    for(unsigned int _k = 0;_k<tracks.size();_k++)
    {
      for(unsigned int _k2=0;_k2<tracks[0].size();_k2++)
      {
        cout<<tracks[_k][_k2]<<",";
      }
      cout<<endl;
    }
  }
  /*   test   */
  
  
  msgs::DetectedObjectArray pub_array;
  pub_array.header = msg->header;
  //copy vector
  // pub_array.objects.assign(msg->objects.begin(),msg->objects.end());

  for (auto obj : msg->objects)
  {
    /******** compare & tracks to msg *********/
    for(unsigned int i=0;i<tracks.size();i++)
    {
      if(obj.camInfo.u == tracks[i][1] &&	obj.camInfo.v == tracks[i][2] && obj.camInfo.width == tracks[i][3] && obj.camInfo.height == tracks[i][4])
      {
        obj.track.id=tracks[i][0];
        pub_array.objects.emplace_back(obj);
      }
    }	

  }

  /********* publish ************/
  chatter_pub_front.publish(pub_array);

  stop = ros::Time::now();
  total_time += stop - start;

#if PRINT_MESSAGE
  std::cout << "Cost time: " << stop - start << " sec" << std::endl;
  std::cout << "Total time: " << total_time << " sec / loop: " << count << std::endl;
#endif
}

void BagToMOT::pedestrian_event()
{
  // AsyncSpinner reference:
  //  https://gist.github.com/bgromov/45ebeced9e8067d9f13cceececc00d5b#file-test_spinner-cpp-L63

  // This node handle uses global callback queue
  ros::NodeHandle nh_sub_1;

  // Set custom callback queue

  ros::Subscriber sub_1;  // nh_sub_1

  sub_1 = nh_sub_1.subscribe("/cam_obj/front_bottom_60", 1, &BagToMOT::chatter_callback,
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

  ped::BagToMOT pe;
  pe.count = 0;
  /************  Initial GMPHD param *****************/
  GMPHDOGMparams sceneParam;
  //setting param
  sceneParam.DET_MIN_CONF = -100;    // Detection Confidence Threshold
  sceneParam.T2TA_MAX_INTERVAL = 80;  // T2TA Maximum Interval
  sceneParam.TRACK_MIN_SIZE = 2;     // Track Minium Length
  sceneParam.FRAMES_DELAY_SIZE = sceneParam.TRACK_MIN_SIZE - 1;
  sceneParam.GROUP_QUEUE_SIZE = sceneParam.TRACK_MIN_SIZE * 10;
  //
  pe.tracker->SetParams(sceneParam);

  ros::NodeHandle nh1;
  pe.chatter_pub_front = nh1.advertise<msgs::DetectedObjectArray>("/Tracking2D/front_bottom_60", 1);
  stop = ros::Time::now();
  std::cout << "PedCross started. Init time: " << stop - start << " sec" << std::endl;
  
  pe.run();
  return 0;
}

