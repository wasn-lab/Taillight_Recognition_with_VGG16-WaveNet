#include "box_fusion.h"

using namespace std;
using namespace DriveNet;

Boxfusion::~Boxfusion()
{

}

Boxfusion::Boxfusion()
{
  front_bottom.LeftLinePoint1 = cv::Point(0, 821);
  front_bottom.LeftLinePoint2 = cv::Point(0, 1207);
  front_bottom.RightLinePoint1 = cv::Point(938, 821);
  front_bottom.RightLinePoint2 = cv::Point(938, 1207);

  left_back.LeftLinePoint1 = cv::Point(1115, 135);
  left_back.LeftLinePoint2 = cv::Point(1115, 340);
  left_back.RightLinePoint1 = cv::Point(1832, 135);
  left_back.RightLinePoint2 = cv::Point(1832, 340);

}

std::vector<msgs::DetectedObjectArray> Boxfusion::boxfuse(std::vector<msgs::DetectedObjectArray> ori_object_arrs, int camera_id_1, int camera_id_2)
{
  bool check_data_1 = false;
  bool check_data_2 = false;

  msgs::DetectedObjectArray object_1, object_2, object_out;
  for(size_t cam_id = 0; cam_id < ori_object_arrs.size(); cam_id++)
  {
    for(const auto& obj : ori_object_arrs[cam_id].objects)
    {
      if(obj.camInfo.id == camera_id_1)check_data_1 = true;
      if(obj.camInfo.id == camera_id_2)check_data_2 = true;  
      // cout << obj.camInfo.u << "," << obj.camInfo.v << endl;    
      
    }
  }

  // Check if these camera have data
  if(!check_data_1 || !check_data_2)
  {    
    return ori_object_arrs;
  }

  // 
  
  for(size_t cam_id = 0; cam_id < ori_object_arrs.size(); cam_id++)
  {
    for(const auto& obj : ori_object_arrs[cam_id].objects)
    {
      if(obj.camInfo.id == camera_id_1)
      {
        object_1.objects.push_back(obj);
      }else if(obj.camInfo.id == camera_id_2)
      {
        object_2.objects.push_back(obj);
      }
    }
  } 

  // cout << "Before box fusion:" << object_2.objects.size() << endl;
  
  // Compare two arrays
  object_out = fusetwocamera(object_1, object_2);

  // cout << "After box fusion:" << object_out.objects.size() << endl;
  

  // Delete original array of left back
  for(size_t cam_id = 0; cam_id < ori_object_arrs.size(); cam_id++)
  {
    if(ori_object_arrs[cam_id].objects[0].camInfo.id == camera_id_2)
    {
      ori_object_arrs.erase(ori_object_arrs.begin() + cam_id);
      break;
    }
  }  
  ori_object_arrs.push_back(object_out);
  
  return ori_object_arrs;
}

msgs::DetectedObjectArray Boxfusion::fusetwocamera(msgs::DetectedObjectArray obj1, msgs::DetectedObjectArray obj2)
{
  msgs::DetectedObjectArray out;
  for(const auto& obj_2 : obj2.objects)
  {
    bool success = false;
    
    for(const auto& obj_1 : obj1.objects)
    {
      if(obj_1.classId != obj_2.classId) continue;
      
      PixelPosition obj1_center, obj2_center;
      std::vector<PixelPosition> bbox_positions1(2);
      bbox_positions1[0].u = obj_1.camInfo.u;
      bbox_positions1[0].v = obj_1.camInfo.v;
      bbox_positions1[1].u = obj_1.camInfo.u + obj_1.camInfo.width;
      bbox_positions1[1].v = obj_1.camInfo.v + obj_1.camInfo.height;
      obj1_center.u = (bbox_positions1[0].u + bbox_positions1[1].u) / 2;
      obj1_center.v = bbox_positions1[1].v;

      std::vector<PixelPosition> bbox_positions2(2);
      bbox_positions2[0].u = obj_2.camInfo.u;
      bbox_positions2[0].v = obj_2.camInfo.v;
      bbox_positions2[1].u = obj_2.camInfo.u + obj_2.camInfo.width;
      bbox_positions2[1].v = obj_2.camInfo.v + obj_2.camInfo.height;
      obj2_center.u = (bbox_positions2[0].u + bbox_positions2[1].u) / 2;
      obj2_center.v = bbox_positions2[1].v;

      // Check these box is overlap or not
      if(CheckPointInArea(front_bottom, obj1_center.u, bbox_positions1[1].v) == 0 
      && CheckPointInArea(left_back, obj2_center.u, bbox_positions2[1].v) == 0)
      {
        PixelPosition obj2_center_trans;        
        
        // Project left back camera to front bottom camera
        obj2_center_trans.u = (((float)obj2_center.u - 1115)/717) * 938;
        obj2_center_trans.v = (((float)obj2_center.v - 135)/205)*386 + 821;

        // cout << obj2_center_trans.u << "," << obj2_center_trans.v << "," << obj1_center.u << "," << obj1_center.v << endl;
        
        // IOU comparison
        if(pointcompare(obj1_center, obj2_center_trans))
        {
          success = true;
          break;
        }       
      }
    }
    if(!success)out.objects.push_back(obj_2);
    
  }
  
  return out;
}

int Boxfusion::CheckPointInArea(CheckArea area, int object_x1, int object_y2)
{
  int point0 = 0;
  int point1 = 1;
  int point2 = 2;
  /// right
  int c1 = (area.RightLinePoint1.x - area.RightLinePoint2.x) * (object_y2 - area.RightLinePoint2.y) -
           (object_x1 - area.RightLinePoint2.x) * (area.RightLinePoint1.y - area.RightLinePoint2.y);
  /// left
  int c2 = (area.LeftLinePoint1.x - area.LeftLinePoint2.x) * (object_y2 - area.LeftLinePoint2.y) -
           (object_x1 - area.LeftLinePoint2.x) * (area.LeftLinePoint1.y - area.LeftLinePoint2.y);

  if (c1 > 0)
  {
    return point2;
  }
  else if (c2 < 0)
  {
    return point1;
  }
  else
  {
    return point0;
  }
}

bool Boxfusion::pointcompare(PixelPosition front_bottom, PixelPosition projected)
{
  // cout << sqrt(pow((front_bottom.u - projected.u), 2) + pow((front_bottom.v - projected.v), 2)) << endl;
  if(sqrt(pow((front_bottom.u - projected.u), 2) + pow((front_bottom.v - projected.v), 2)) < pixelthres)
  {
    return true;
  }
  return false;
}