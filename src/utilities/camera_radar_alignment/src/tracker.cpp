#include "tracker.h"

msgs::DetectedObjectArray Tracker::tracking(msgs::DetectedObjectArray camera_objects)
{
  msgs::DetectedObjectArray result(camera_objects);
  if(kalman_vector.empty())
  {
    for(std::size_t i = 0; i < result.objects.size(); i++)
    {
      cv::Rect rect = cv::Rect(result.objects[i].camInfo[0].u, result.objects[i].camInfo[0].v, 
                           result.objects[i].camInfo[0].width, result.objects[i].camInfo[0].height);
      Kalman kf(tracking_id);
      kf.update(rect, true);
      kalman_vector.push_back(kf);
      result.objects[i].camInfo[0].id = tracking_id;
      tracking_id++;
    }
  }
  else
  {
    //matching by iou & update or creat kalman filter
    for(std::size_t i = 0; i < result.objects.size(); i++)
    {
      cv::Rect rect = cv::Rect(result.objects[i].camInfo[0].u, result.objects[i].camInfo[0].v, 
                           result.objects[i].camInfo[0].width, result.objects[i].camInfo[0].height);
      double iou_max = -1;
      int k = -1;
      for(std::size_t j = 0; j < kalman_vector.size(); j++)
      {
        cv::Rect rect1 = rect & kalman_vector.at(j).last_detection;
        float rect2 = rect.area() + kalman_vector.at(j).last_detection.area() - rect1.area();
        double iou = (double)rect1.area() / rect2;
        if(iou > 0.3 && iou > iou_max)
        {
          iou_max = iou;
          k = j;
        }
      }
      if(k != -1)
      {
        kalman_vector.at(k).predict();
        kalman_vector.at(k).isUpdated = true;
        cv::Rect currected_rect = kalman_vector.at(k).update(rect, true);
        result.objects[i].camInfo[0].id = kalman_vector.at(k).id;
        result.objects[i].camInfo[0].u = currected_rect.x;
        result.objects[i].camInfo[0].v = currected_rect.y;
        result.objects[i].camInfo[0].width = currected_rect.width ;
        result.objects[i].camInfo[0].height = currected_rect.height;
      }
      else
      {
        if(kalman_vector.size() < max_tracking_num)
        {
          Kalman kf(tracking_id);
          kf.update(rect, true);
          kalman_vector.push_back(kf);
          result.objects[i].camInfo[0].id = tracking_id;
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
      }
      else
      {
        //kalman_vector.at(i).predict();
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
  }
  return result;
}
