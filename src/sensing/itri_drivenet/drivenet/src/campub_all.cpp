#include "drivenet/campub.h"

void callback_CamObjFC(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  HeaderAll = DetectMsg->header;
  arrCamObjFC = DetectMsg->objects;
}

void callback_CamObjFTf(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  arrCamObjFTf = DetectMsg->objects;
}

void callback_CamObjFTc(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  arrCamObjFTc = DetectMsg->objects;
}

void callback_CamObjRF(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  arrCamObjRF = DetectMsg->objects;
}

void callback_CamObjRB(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  arrCamObjRB = DetectMsg->objects;
}

void callback_CamObjLF(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  arrCamObjLF = DetectMsg->objects;
}

void callback_CamObjLB(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  arrCamObjLB = DetectMsg->objects;
}

void callback_CamObjBT(const msgs::DetectedObjectArray::ConstPtr& DetectMsg)
{
  arrCamObjBT = DetectMsg->objects;
}

void collectRepub()
{
  // arrCamObjAll = arrCamObjBT + arrCamObjLB;
  msgs::DetectedObjectArray arrCamObjAll;
  size_t allSize = arrCamObjFC.size() + arrCamObjFTf.size() + arrCamObjFTc.size() +
                   arrCamObjRF.size() + arrCamObjRB.size() + arrCamObjLF.size() + arrCamObjLB.size() +
                   arrCamObjBT.size();
  arrCamObjAll.objects.reserve(allSize);
  arrCamObjAll.header = HeaderAll;

  for (size_t i = 0; i < arrCamObjFC.size(); i++)
  {
    arrCamObjFC[i].camInfo.id = camera::id::front_bottom_60;
    arrCamObjAll.objects.push_back(arrCamObjFC[i]);
  }

  for (size_t i = 0; i < arrCamObjFTf.size(); i++)
  {
    arrCamObjFTf[i].camInfo.id = camera::id::front_top_far_30;
    arrCamObjAll.objects.push_back(arrCamObjFTf[i]);
  }

  for (size_t i = 0; i < arrCamObjFTc.size(); i++)
  {
    arrCamObjFTc[i].camInfo.id = camera::id::front_top_close_120;
    arrCamObjAll.objects.push_back(arrCamObjFTc[i]);
  }

  for (size_t i = 0; i < arrCamObjRF.size(); i++)
  {
    arrCamObjRF[i].camInfo.id = camera::id::right_front_60;
    arrCamObjAll.objects.push_back(arrCamObjRF[i]);
  }

  for (size_t i = 0; i < arrCamObjRB.size(); i++)
  {
    arrCamObjRB[i].camInfo.id = camera::id::right_back_60;
    arrCamObjAll.objects.push_back(arrCamObjRB[i]);
  }

  for (size_t i = 0; i < arrCamObjLF.size(); i++)
  {
    arrCamObjLF[i].camInfo.id = camera::id::left_front_60;
    arrCamObjAll.objects.push_back(arrCamObjLF[i]);
  }

  for (size_t i = 0; i < arrCamObjLB.size(); i++)
  {
    arrCamObjLB[i].camInfo.id = camera::id::left_back_60;
    arrCamObjAll.objects.push_back(arrCamObjLB[i]);
  }

  for (size_t i = 0; i < arrCamObjBT.size(); i++)
  {
    arrCamObjBT[i].camInfo.id = camera::id::back_top_120;
    arrCamObjAll.objects.push_back(arrCamObjBT[i]);
  }

  ROS_INFO_STREAM("HEADER: " << arrCamObjAll.header);

  CamObjAll.publish(arrCamObjAll);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "CamPub");
  ros::NodeHandle nh;

  // cam60_0_topicName = camera::topics[camera::id::right_60];
  // cam60_0 = nh.subscribe(cam60_0_topicName + std::string("/compressed"), 1, callback_60_0_decode);

  CamObjFC_topicName = camera::topics_obj[camera::id::front_bottom_60];
  CamObjFTf_topicName = camera::topics_obj[camera::id::front_top_far_30];
  CamObjFTc_topicName = camera::topics_obj[camera::id::front_top_close_120];
  CamObjRF_topicName = camera::topics_obj[camera::id::right_front_60];
  CamObjRB_topicName = camera::topics_obj[camera::id::right_back_60];
  CamObjLF_topicName = camera::topics_obj[camera::id::left_front_60];
  CamObjLB_topicName = camera::topics_obj[camera::id::left_back_60];
  CamObjBT_topicName = camera::topics_obj[camera::id::back_top_120];

  // Subscribe msgs
  CamObjFC = nh.subscribe(CamObjFC_topicName, 1, callback_CamObjFC);
  CamObjFTf = nh.subscribe(CamObjFTf_topicName, 1, callback_CamObjFTf);
  CamObjFTc = nh.subscribe(CamObjFTc_topicName, 1, callback_CamObjFTc);
  CamObjRF = nh.subscribe(CamObjRF_topicName, 1, callback_CamObjRF);
  CamObjRB = nh.subscribe(CamObjRB_topicName, 1, callback_CamObjRB);
  CamObjLF = nh.subscribe(CamObjLF_topicName, 1, callback_CamObjLF);
  CamObjLB = nh.subscribe(CamObjLB_topicName, 1, callback_CamObjLB);
  CamObjBT = nh.subscribe(CamObjBT_topicName, 1, callback_CamObjBT);

  CamObjAll = nh.advertise<msgs::DetectedObjectArray>(camera::detect_result, 8);

  ros::Rate loop_rate(30);

  while (ros::ok())
  {
    collectRepub();
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
