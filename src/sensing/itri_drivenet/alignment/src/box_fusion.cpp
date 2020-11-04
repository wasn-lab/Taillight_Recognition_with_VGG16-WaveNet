#include "box_fusion.h"

using namespace std;
using namespace DriveNet;

Boxfusion::~Boxfusion()
{
}

Boxfusion::Boxfusion()
{
  front_bottom.LeftLinePoint1 = cv::Point(FB_left_top_x, FB_left_top_y);
  front_bottom.LeftLinePoint2 = cv::Point(FB_left_top_x, FB_right_bottom_y);
  front_bottom.RightLinePoint1 = cv::Point(FB_right_bottom_x, FB_left_top_y);
  front_bottom.RightLinePoint2 = cv::Point(FB_right_bottom_x, FB_right_bottom_y);

  left_back.LeftLinePoint1 = cv::Point(LB_left_top_x, LB_left_top_y);
  left_back.LeftLinePoint2 = cv::Point(LB_left_top_x, LB_right_bottom_y);
  left_back.RightLinePoint1 = cv::Point(LB_right_bottom_x, LB_left_top_y);
  left_back.RightLinePoint2 = cv::Point(LB_right_bottom_x, LB_right_bottom_y);
}

enum checkBoxStatus
{
  InsideArea,
  OverLeftBound,
  OverRightBound
};

std::vector<msgs::DetectedObject> Boxfusion::multiCamBoxFuse(std::vector<msgs::DetectedObject> input_obj_arrs)
{

  auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
    if (x1min > x2min)
    {
      std::swap(x1min, x2min);
      std::swap(x1max, x2max);
    }
    return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
  };
  auto computeIoU = [&overlap1D](msgs::DetectedObject obj1, msgs::DetectedObject obj2) -> float {
    float overlapX = overlap1D(obj1.bPoint.p0.x, obj1.bPoint.p7.x, obj2.bPoint.p0.x, obj2.bPoint.p7.x);
    float overlapY = overlap1D(obj1.bPoint.p0.y, obj1.bPoint.p7.y, obj2.bPoint.p0.y, obj2.bPoint.p7.y);
    float area1 = (obj1.bPoint.p7.x - obj1.bPoint.p0.x) * (obj1.bPoint.p7.y - obj1.bPoint.p0.y);
    float area2 = (obj2.bPoint.p7.x - obj2.bPoint.p0.x) * (obj2.bPoint.p7.y - obj2.bPoint.p0.y);
    float overlap2D = overlapX * overlapY;
    float u = area1 + area2 - overlap2D;
    return u == 0 ? 0 : overlap2D / u;
  };

  std::vector<msgs::DetectedObject> input_copy1;

  input_copy1.assign(input_obj_arrs.begin(), input_obj_arrs.end());
  
  // std::cout << "Before:" << input_copy1.size() << std::endl;        
  
  // Compare 3D bounding boxes
  for(uint i = 0; i < input_copy1.size(); i++)
  {
    for(uint j = 0; j < input_copy1.size(); j++)
    {
      if(input_copy1[j].classId != input_copy1[i].classId || 
      /*input_copy1[i].camInfo.id == input_copy1[j].camInfo.id ||*/
      input_copy1[j].classId == 99 || input_copy1[i].classId == 99 || i == j)
      {
        continue;
      }
      float overlap = computeIoU(input_copy1[i], input_copy1[j]);
      if(overlap > iou_threshold)
      {
        input_copy1[j].classId = 99;
      }
    }
  }

  std::vector<msgs::DetectedObject> output;
  
  // Delete box
  for(uint a = 0; a < input_copy1.size(); a++)
  {
    if(input_copy1[a].classId != 99)
    {
      output.push_back(input_copy1[a]);
    }
  }

  // std::cout << "After:" << output.size() << std::endl;        
  return output;
}

std::vector<msgs::DetectedObjectArray> Boxfusion::boxfuse(std::vector<msgs::DetectedObjectArray> ori_object_arrs,
                                                          int camera_id_1, int camera_id_2)
{
  bool check_data_1 = false;
  bool check_data_2 = false;

  msgs::DetectedObjectArray object_1, object_2, object_out;
  for (size_t cam_id = 0; cam_id < ori_object_arrs.size(); cam_id++)
  {
    for (const auto& obj : ori_object_arrs[cam_id].objects)
    {
      if (obj.camInfo.id == camera_id_1)
      {
        check_data_1 = true;
      }
      if (obj.camInfo.id == camera_id_2)
      {
        check_data_2 = true;
      }
      // cout << obj.camInfo.u << "," << obj.camInfo.v << endl;
    }
  }

  // Check if these camera have data
  if (!check_data_1 || !check_data_2)
  {
    return ori_object_arrs;
  }

  //

  for (size_t cam_id = 0; cam_id < ori_object_arrs.size(); cam_id++)
  {
    for (const auto& obj : ori_object_arrs[cam_id].objects)
    {
      if (obj.camInfo.id == camera_id_1)
      {
        object_1.objects.push_back(obj);
      }
      else if (obj.camInfo.id == camera_id_2)
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
  for (size_t cam_id = 0; cam_id < ori_object_arrs.size(); cam_id++)
  {
    if (ori_object_arrs[cam_id].objects[0].camInfo.id == camera_id_2)
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
  for (const auto& obj_2 : obj2.objects)
  {
    bool success = false;

    for (const auto& obj_1 : obj1.objects)
    {
      if (obj_1.classId != obj_2.classId)
      {
        continue;
      }

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
      if (CheckPointInArea(front_bottom, obj1_center.u, bbox_positions1[1].v) == 0 &&
          CheckPointInArea(left_back, obj2_center.u, bbox_positions2[1].v) == 0)
      {
        PixelPosition obj2_center_trans;

        // Project left back camera to front bottom camera
        obj2_center_trans.u =
            (((float)obj2_center.u - LB_left_top_x) / (LB_right_bottom_x - LB_left_top_x)) * FB_right_bottom_x;
        obj2_center_trans.v = (((float)obj2_center.v - LB_left_top_y) / (LB_right_bottom_y - LB_left_top_y)) *
                                  (FB_right_bottom_y - FB_left_top_y) +
                              FB_left_top_y;

        // cout << obj2_center_trans.u << "," << obj2_center_trans.v << "," << obj1_center.u << "," << obj1_center.v <<
        // endl;

        // IOU comparison
        if (pointcompare(obj1_center, obj2_center_trans))
        {
          success = true;
          break;
        }
      }
    }
    if (!success)
    {
      out.objects.push_back(obj_2);
    }
  }

  return out;
}

int Boxfusion::CheckPointInArea(CheckArea area, int object_x1, int object_y2)
{
  /// right
  int c1 = (area.RightLinePoint1.x - area.RightLinePoint2.x) * (object_y2 - area.RightLinePoint2.y) -
           (object_x1 - area.RightLinePoint2.x) * (area.RightLinePoint1.y - area.RightLinePoint2.y);
  /// left
  int c2 = (area.LeftLinePoint1.x - area.LeftLinePoint2.x) * (object_y2 - area.LeftLinePoint2.y) -
           (object_x1 - area.LeftLinePoint2.x) * (area.LeftLinePoint1.y - area.LeftLinePoint2.y);

  if (c1 > 0)
  {
    return checkBoxStatus::OverRightBound;
  }
  else if (c2 < 0)
  {
    return checkBoxStatus::OverLeftBound;
  }
  else
  {
    return checkBoxStatus::InsideArea;
  }
}

bool Boxfusion::pointcompare(PixelPosition front_bottom, PixelPosition projected)
{
  return bool(sqrt(pow((front_bottom.u - projected.u), 2) + pow((front_bottom.v - projected.v), 2)) < pixelthres);
}