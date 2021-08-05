#include "tracker.h"

msgs::DetectedObjectArray Tracker::tracking(msgs::DetectedObjectArray camera_objects)
{
  msgs::DetectedObjectArray result(std::move(camera_objects));
  if(kalman_vector.empty())
  {
    for(auto & object : result.objects)
    {
      cv::Rect rect = cv::Rect(object.camInfo[0].u, object.camInfo[0].v, 
                           object.camInfo[0].width, object.camInfo[0].height);
      int type = object.classId;
      Kalman kf(tracking_id, type);
      kf.update(rect, true);
      kalman_vector.push_back(kf);
      object.camInfo[0].id = tracking_id;
      tracking_id++;
    }
  }
  else
  {
    //matching by iou & update or creat kalman filter
    std::vector<std::vector<double>> iou_table(result.objects.size(), std::vector<double>(kalman_vector.size()));
    for(auto & object : result.objects)
    {
      cv::Rect rect = cv::Rect(object.camInfo[0].u, object.camInfo[0].v, 
                           object.camInfo[0].width, object.camInfo[0].height);
      int type = object.classId;
      double iou_max = -1;
      int k = -1;
      for(std::size_t j = 0; j < kalman_vector.size(); j++)
      {
        cv::Rect rect1 = rect & kalman_vector.at(j).last_detection;
        float rect2 = rect.area() + kalman_vector.at(j).last_detection.area() - rect1.area();
        double iou = (double)rect1.area() / rect2;
        //iou_table.at(i).at(j) = iou;
        if (type == kalman_vector.at(j).type)
        {
          if(rect.area() > 2000 && iou > 0.2 && iou > iou_max)
          {
            iou_max = iou;
            k = j;
          }
          else if(rect.area() > 1000 && iou > 0.25 && iou > iou_max)
          {
            iou_max = iou;
            k = j;
          }
          else if(rect.area() > 500 && iou > 0.3 && iou > iou_max)
          {
            iou_max = iou;
            k = j;
          }
          else if(rect.area() > 400 && iou > 0.325 && iou > iou_max)
          {
            iou_max = iou;
            k = j;
          }
          else if(rect.area() > 250 && iou > 0.35 && iou > iou_max)
          {
            iou_max = iou;
            k = j;
          }
          else if(iou > 0.4 && iou > iou_max)
          {
            iou_max = iou;
            k = j;
          }
        }
      }
      if(k != -1)
      {
        kalman_vector.at(k).predict();
        kalman_vector.at(k).isUpdated = true;
        cv::Rect currected_rect = kalman_vector.at(k).update(rect, true);
        object.camInfo[0].id = kalman_vector.at(k).id;
        object.camInfo[0].u = currected_rect.x;
        object.camInfo[0].v = currected_rect.y;
        object.camInfo[0].width = currected_rect.width ;
        object.camInfo[0].height = currected_rect.height;
      }
      else
      {
        if(kalman_vector.size() < (unsigned)max_tracking_num)
        {
          std::cout << "new object: " << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << " " << rect.area() << std::endl;
          Kalman kf(tracking_id, type);
          kf.update(rect, true);
          kalman_vector.push_back(kf);
          object.camInfo[0].id = tracking_id;
          tracking_id++;
        }
      }
    }
    //update no matching object & remove lost object
    for(std::size_t i = 0; i < kalman_vector.size(); i++)
    {
      if(kalman_vector.at(i).tracking_count == max_tracking_frames)
      {
        kalman_vector.erase(kalman_vector.begin() + i);
        i--;
        continue;
      }
      if(kalman_vector.at(i).isUpdated)
      {
        kalman_vector.at(i).isUpdated = false;
        kalman_vector.at(i).max_iou = -1;
      }
      else
      {
        double center_x = kalman_vector.at(i).last_detection.x + (kalman_vector.at(i).last_detection.width / 2.0);
        double center_y = kalman_vector.at(i).last_detection.y + (kalman_vector.at(i).last_detection.height / 2.0);
        if(center_x < 50 || center_y < 50 || center_x > 1230 || center_y > 670)
        {
          kalman_vector.erase(kalman_vector.begin() + i);
          i--;
          continue;
        }
        kalman_vector.at(i).predict();
        kalman_vector.at(i).max_iou = -1;
        cv::Rect rect_null;
        cv::Rect tracking_rect = kalman_vector.at(i).update(rect_null, false);
        msgs::CamInfo caminfo;
        caminfo.id = kalman_vector.at(i).id;
        caminfo.u = tracking_rect.x;
        caminfo.v = tracking_rect.y;
        caminfo.width = tracking_rect.width ;
        caminfo.height = tracking_rect.height;
        msgs::DetectedObject object;
        object.camInfo.push_back(caminfo);
        result.objects.push_back(object);
      }
    }
        std::cout << std::endl;
  }
  return result;
}
