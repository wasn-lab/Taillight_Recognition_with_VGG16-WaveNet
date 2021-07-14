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

  cam_info_vector = std::vector<msgs::CamInfo>(camera::id::num_ids);
  cam_info_vector_tmp = std::vector<msgs::CamInfo>(camera::id::num_ids);
  
  for (size_t id = 0; id < camera::id::num_ids; id++)
  {
    msgs::CamInfo cam_info;
    cam_info.u = 0;
    cam_info.v = 0;
    cam_info.width = 0;
    cam_info.height = 0;
    cam_info.prob = -1;
    cam_info.id = id;
    cam_info_vector[id] = cam_info;
    cam_info_vector_tmp[id] = cam_info;
  }
}

enum checkBoxStatus
{
  InsideArea,
  OverLeftBound,
  OverRightBound
};

enum overlapped
{
  OverLapped = 99
};

std::vector<msgs::DetectedObject> Boxfusion::multi_cambox_fuse(std::vector<msgs::DetectedObject>& input_obj_arrs)
{
  auto overlap_1d = [](float x1min, float x1max, float x2min, float x2max) -> float {
    if (x1min > x2min)
    {
      std::swap(x1min, x2min);
      std::swap(x1max, x2max);
    }
    return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
  };
  auto compute_iou = [&overlap_1d](msgs::DetectedObject& obj1, msgs::DetectedObject& obj2) -> float {
    float overlap_x = overlap_1d(obj1.bPoint.p0.x, obj1.bPoint.p6.x, obj2.bPoint.p0.x, obj2.bPoint.p6.x);
    float overlap_y = overlap_1d(obj1.bPoint.p0.y, obj1.bPoint.p6.y, obj2.bPoint.p0.y, obj2.bPoint.p6.y);
    float area1 = abs(obj1.bPoint.p6.x - obj1.bPoint.p0.x) * abs(obj1.bPoint.p6.y - obj1.bPoint.p0.y);
    float area2 = abs(obj2.bPoint.p6.x - obj2.bPoint.p0.x) * abs(obj2.bPoint.p6.y - obj2.bPoint.p0.y);
    float overlap_2d = overlap_x * overlap_y;
    float u = area1 + area2 - overlap_2d;

    return u == 0 ? 0 : overlap_2d / u;
  };

  std::vector<msgs::DetectedObject> input_copy1;
  std::vector<msgs::DetectedObject> no_oriented;

  input_copy1.assign(input_obj_arrs.begin(), input_obj_arrs.end());
  no_oriented.assign(input_obj_arrs.begin(), input_obj_arrs.end());

  for (uint a = 0; a < input_copy1.size(); a++)
  {
    no_oriented[a].bPoint = redefine_bounding_box(input_copy1[a].bPoint);
  }

  // Compare 3D bounding boxes
  for (uint i = 0; i < input_copy1.size(); i++)
  {
    for (uint j = 0; j < input_copy1.size(); j++)
    {
      bool remove_new_obj = false;
      if (input_copy1[j].classId == overlapped::OverLapped || input_copy1[i].classId == overlapped::OverLapped ||
          i == j)
      {
        continue;
      }
      // Algo 1: Use no heading
      float overlap = abs(compute_iou(no_oriented[i], no_oriented[j]));

      // Algo 2: Use IoU comparison with heading(still working)
      // float overlap2 = iou_compare_with_heading(input_copy1[i], input_copy1[j]);

      if (overlap == 1)
      {
        if (input_copy1[i].classId == static_cast<int>(DriveNet::common_type_id::person) && input_copy1[j].classId == static_cast<int>(DriveNet::common_type_id::motorbike))
        {
          input_copy1[i].classId = static_cast<int>(DriveNet::common_type_id::motorbike);
        }
        else if (input_copy1[i].classId == static_cast<int>(DriveNet::common_type_id::person) && input_copy1[j].classId == static_cast<int>(DriveNet::common_type_id::bicycle))
        {
          input_copy1[i].classId = static_cast<int>(DriveNet::common_type_id::bicycle);
        }

        if (input_copy1[i].camInfo[0].id == input_copy1[j].camInfo[0].id)
        {
          if ((input_copy1[i].classId == static_cast<int>(DriveNet::common_type_id::motorbike) || input_copy1[i].classId == static_cast<int>(DriveNet::common_type_id::bicycle))
          && input_copy1[j].classId == static_cast<int>(DriveNet::common_type_id::person))
          {
            remove_new_obj = true;
          }
        }

        cam_info_vector_tmp = cam_info_vector;
        if (input_copy1[i].camInfo.size() == camera::id::num_ids)
        {
          cam_info_vector_tmp = input_copy1[i].camInfo;
        }
        else
        {
          cam_info_vector_tmp[input_copy1[i].camInfo[0].id] = input_copy1[i].camInfo[0];
        }
        if (!remove_new_obj)
        {
          if (input_copy1[j].camInfo.size() == camera::id::num_ids)
          {
            for (size_t id = 0; id < camera::id::num_ids; id++)
            {
              cam_info_vector_tmp[id] = input_copy1[j].camInfo[id];
            }
          }
          else
          {
            cam_info_vector_tmp[input_copy1[j].camInfo[0].id] = input_copy1[j].camInfo[0];
          }
        }
        
        input_copy1[i].camInfo = cam_info_vector_tmp;
        input_copy1[j].classId = overlapped::OverLapped; // Clear box
      }
      else
      {
        if (input_copy1[i].camInfo.size() == camera::id::num_ids)
        {
          cam_info_vector_tmp = input_copy1[i].camInfo;
        }
        else
        {
          cam_info_vector_tmp[input_copy1[i].camInfo[0].id] = input_copy1[i].camInfo[0];
        }
        input_copy1[i].camInfo = cam_info_vector_tmp;

        cam_info_vector_tmp = cam_info_vector;
        if (input_copy1[j].camInfo.size() == camera::id::num_ids)
        {
          cam_info_vector_tmp = input_copy1[j].camInfo;
        }
        else
        {
          cam_info_vector_tmp[input_copy1[j].camInfo[0].id] = input_copy1[j].camInfo[0];
        }
        input_copy1[j].camInfo = cam_info_vector_tmp;
      }
    }
  }

  std::vector<msgs::DetectedObject> output;

  // Delete box
  for (uint a = 0; a < input_copy1.size(); a++)
  {
    if (input_copy1[a].classId != overlapped::OverLapped)
    {
      output.push_back(input_copy1[a]);
    }
  }

  return output;
}

msgs::BoxPoint Boxfusion::redefine_bounding_box(msgs::BoxPoint origin_box)
{
  msgs::BoxPoint processed_box;
  msgs::PointXYZ max_box_point, min_box_point;

  if (origin_box.p0.x > origin_box.p6.x)
  {
    max_box_point.x = origin_box.p0.x;
    min_box_point.x = origin_box.p6.x;
  }
  else
  {
    min_box_point.x = origin_box.p0.x;
    max_box_point.x = origin_box.p6.x;
  }

  if (origin_box.p0.y > origin_box.p6.y)
  {
    max_box_point.y = origin_box.p0.y;
    min_box_point.y = origin_box.p6.y;
  }
  else
  {
    min_box_point.y = origin_box.p0.y;
    max_box_point.y = origin_box.p6.y;
  }

  if (origin_box.p0.z > origin_box.p6.z)
  {
    max_box_point.z = origin_box.p0.z;
    min_box_point.z = origin_box.p6.z;
  }
  else
  {
    min_box_point.z = origin_box.p0.z;
    max_box_point.z = origin_box.p6.z;
  }

  processed_box.p0.x = min_box_point.x;
  processed_box.p0.y = min_box_point.y;
  processed_box.p0.z = min_box_point.z;

  processed_box.p1.x = min_box_point.x;
  processed_box.p1.y = min_box_point.y;
  processed_box.p1.z = max_box_point.z;

  processed_box.p2.x = max_box_point.x;
  processed_box.p2.y = min_box_point.y;
  processed_box.p2.z = max_box_point.z;

  processed_box.p3.x = max_box_point.x;
  processed_box.p3.y = min_box_point.y;
  processed_box.p3.z = min_box_point.z;

  processed_box.p4.x = min_box_point.x;
  processed_box.p4.y = max_box_point.y;
  processed_box.p4.z = min_box_point.z;

  processed_box.p5.x = min_box_point.x;
  processed_box.p5.y = max_box_point.y;
  processed_box.p5.z = max_box_point.z;

  processed_box.p6.x = max_box_point.x;
  processed_box.p6.y = max_box_point.y;
  processed_box.p6.z = max_box_point.z;

  processed_box.p7.x = max_box_point.x;
  processed_box.p7.y = max_box_point.y;
  processed_box.p7.z = min_box_point.z;

  return processed_box;
}

float Boxfusion::iou_compare_with_heading(msgs::DetectedObject& obj1, msgs::DetectedObject& obj2)
{
  std::cout << "@@@@@" << std::endl;
  std::cout << obj1.bPoint.p0.x * 100 << "," << obj1.bPoint.p3.x * 100 << "," << obj1.bPoint.p7.x * 100 << ","
            << obj1.bPoint.p0.y * 100 << "," << obj1.bPoint.p3.y * 100 << "," << obj1.bPoint.p7.y * 100 << ","
            << std::endl;
  std::cout << obj2.bPoint.p0.x * 100 << "," << obj2.bPoint.p3.x * 100 << "," << obj2.bPoint.p7.x * 100 << ","
            << obj2.bPoint.p0.y * 100 << "," << obj2.bPoint.p3.y * 100 << "," << obj2.bPoint.p7.y * 100 << ","
            << std::endl;

  cv::RotatedRect rect1 =
      cv::RotatedRect(cv::Point2f(cvRound(obj1.bPoint.p0.x * 100), cvRound(obj1.bPoint.p0.y * 100)),
                      cv::Point2f(cvRound(obj1.bPoint.p3.x * 100), cvRound(obj1.bPoint.p3.y * 100)),
                      cv::Point2f(cvRound(obj1.bPoint.p7.x * 100), cvRound(obj1.bPoint.p7.y * 100)));

  std::cout << "@@@@@" << std::endl;

  cv::RotatedRect rect2 =
      cv::RotatedRect(cv::Point2f(cvRound(obj2.bPoint.p0.x * 100), cvRound(obj2.bPoint.p0.y * 100)),
                      cv::Point2f(cvRound(obj2.bPoint.p3.x * 100), cvRound(obj2.bPoint.p3.y * 100)),
                      cv::Point2f(cvRound(obj2.bPoint.p7.x * 100), cvRound(obj2.bPoint.p7.y * 100)));

  // float calcIOU(cv::RotatedRect rect1, cv::RotatedRect rect2) {
  std::cout << "#########" << std::endl;

  float area_rect1 = rect1.size.width * rect1.size.height;
  float area_rect2 = rect2.size.width * rect2.size.height;
  vector<cv::Point2f> vertices;

  cv::rotatedRectangleIntersection(rect1, rect2, vertices);
  if (vertices.empty())
  {
    return 0.0;
  }
  else
  {
    vector<cv::Point2f> order_pts;
    // 找到交集（交集的区域），对轮廓的各个点进行排序

    cv::convexHull(cv::Mat(vertices), order_pts, true);
    double area = cv::contourArea(order_pts);
    auto inner = (float)(area / (area_rect1 + area_rect2 - area + 0.0001));
    std::cout << inner << std::endl;
    return inner;
  }
}

std::vector<msgs::DetectedObjectArray> Boxfusion::box_fuse(std::vector<msgs::DetectedObjectArray> ori_object_arrs,
                                                           int camera_id_1, int camera_id_2)
{
  bool check_data_1 = false;
  bool check_data_2 = false;

  msgs::DetectedObjectArray object_1, object_2, object_out;
  // for (size_t cam_id = 0; cam_id < ori_object_arrs.size(); cam_id++)
  for (auto& object_arrs : ori_object_arrs)
  {
    for (const auto& obj : object_arrs.objects)
    {
      if (obj.camInfo[0].id == camera_id_1)
      {
        check_data_1 = true;
      }
      if (obj.camInfo[0].id == camera_id_2)
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

  // for (size_t cam_id = 0; cam_id < ori_object_arrs.size(); cam_id++)
  for (auto& object_arrs : ori_object_arrs)
  {
    for (const auto& obj : object_arrs.objects)
    {
      if (obj.camInfo[0].id == camera_id_1)
      {
        object_1.objects.push_back(obj);
      }
      else if (obj.camInfo[0].id == camera_id_2)
      {
        object_2.objects.push_back(obj);
      }
    }
  }

  // cout << "Before box fusion:" << object_2.objects.size() << endl;

  // Compare two arrays
  object_out = fuse_two_camera(object_1, object_2);

  // cout << "After box fusion:" << object_out.objects.size() << endl;

  // Delete original array of left back
  for (size_t cam_id = 0; cam_id < ori_object_arrs.size(); cam_id++)
  {
    if (ori_object_arrs[cam_id].objects[0].camInfo[0].id == camera_id_2)
    {
      ori_object_arrs.erase(ori_object_arrs.begin() + cam_id);
      break;
    }
  }
  ori_object_arrs.push_back(object_out);

  return ori_object_arrs;
}

msgs::DetectedObjectArray Boxfusion::fuse_two_camera(msgs::DetectedObjectArray obj1, msgs::DetectedObjectArray obj2)
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

      PixelPosition obj1_center{ -1, -1 };
      PixelPosition obj2_center{ -1, -1 };
      std::vector<PixelPosition> bbox_positions1(2);
      bbox_positions1[0].u = obj_1.camInfo[0].u;
      bbox_positions1[0].v = obj_1.camInfo[0].v;
      bbox_positions1[1].u = obj_1.camInfo[0].u + obj_1.camInfo[0].width;
      bbox_positions1[1].v = obj_1.camInfo[0].v + obj_1.camInfo[0].height;
      obj1_center.u = (bbox_positions1[0].u + bbox_positions1[1].u) / 2;
      obj1_center.v = bbox_positions1[1].v;

      std::vector<PixelPosition> bbox_positions2(2);
      bbox_positions2[0].u = obj_2.camInfo[0].u;
      bbox_positions2[0].v = obj_2.camInfo[0].v;
      bbox_positions2[1].u = obj_2.camInfo[0].u + obj_2.camInfo[0].width;
      bbox_positions2[1].v = obj_2.camInfo[0].v + obj_2.camInfo[0].height;
      obj2_center.u = (bbox_positions2[0].u + bbox_positions2[1].u) / 2;
      obj2_center.v = bbox_positions2[1].v;

      // Check these box is overlap or not
      if (check_point_in_area(front_bottom, obj1_center.u, bbox_positions1[1].v) == 0 &&
          check_point_in_area(left_back, obj2_center.u, bbox_positions2[1].v) == 0)
      {
        PixelPosition obj2_center_trans{ -1, -1 };

        // Project left back camera to front bottom camera
        obj2_center_trans.u =
            (((float)obj2_center.u - LB_left_top_x) / (LB_right_bottom_x - LB_left_top_x)) * FB_right_bottom_x;
        obj2_center_trans.v = (((float)obj2_center.v - LB_left_top_y) / (LB_right_bottom_y - LB_left_top_y)) *
                                  (FB_right_bottom_y - FB_left_top_y) +
                              FB_left_top_y;

        // cout << obj2_center_trans.u << "," << obj2_center_trans.v << "," << obj1_center.u << "," << obj1_center.v <<
        // endl;

        // IOU comparison
        if (point_compare(obj1_center, obj2_center_trans))
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

int Boxfusion::check_point_in_area(CheckArea area, int object_x1, int object_y2)
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

bool Boxfusion::point_compare(PixelPosition front_bottom, PixelPosition projected)
{
  return bool(sqrt(pow((front_bottom.u - projected.u), 2) + pow((front_bottom.v - projected.v), 2)) < pixelthres_);
}