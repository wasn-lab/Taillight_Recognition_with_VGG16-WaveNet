#include "kalman.h"

Kalman::Kalman(int id) : id(id)
{
}

void Kalman::init(cv::Rect rect)
{
  KF.init(stateNum, measureNum, 0);
  //轉移矩陣A
  KF.transitionMatrix = (cv::Mat_<float>(8, 8) <<
                                        1, 0, 0, 0, delta_t, 0,       0,       0,
                                        0, 1, 0, 0, 0,       delta_t, 0,       0,
                                        0, 0, 1, 0, 0,       0,       delta_t, 0,
                                        0, 0, 0, 1, 0,       0,       0,       delta_t,
                                        0, 0, 0, 0, 1,       0,       0,       0,
                                        0, 0, 0, 0, 0,       1,       0,       0,
                                        0, 0, 0, 0, 0,       0,       1,       0,
                                        0, 0, 0, 0, 0,       0,       0,       1); 
  float x = rect.x + (rect.width / 2);
  float y = rect.y + (rect.height / 2);
  //初始狀態值x(0)
  KF.statePre.at<float>(0) = x;      // x
  KF.statePre.at<float>(1) = y;      // y
  KF.statePre.at<float>(2) = rect.width;  // width
  KF.statePre.at<float>(3) = rect.height; // height
  KF.statePre.at<float>(4) = 0;     // dx
  KF.statePre.at<float>(5) = 0;     // dy
  KF.statePre.at<float>(6) = 0;            // dw
  KF.statePre.at<float>(7) = 0;            // dh
  //初始測量值x'(0)
  KF.statePost.at<float>(0) = x;
  KF.statePost.at<float>(1) = y;
  KF.statePost.at<float>(2) = rect.width;
  KF.statePost.at<float>(3) = rect.height;
  KF.statePost.at<float>(4) = 0;
  KF.statePost.at<float>(5) = 0;
  KF.statePost.at<float>(6) = 0;
  KF.statePost.at<float>(7) = 0;
  //測量矩陣H
  cv::setIdentity(KF.measurementMatrix);              
  //系統噪聲方差矩陣Q                               
  cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));     
  //測量噪聲方差矩陣R                       
  cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));   
  //後驗錯誤估計協方差矩陣P                     
  cv::setIdentity(KF.errorCovPost, cv::Scalar::all(.1));      
  initialized = true;    
  last_detection = rect;
}

void Kalman::predict()
{
  if (initialized)
  {
    cv::Mat prediction = KF.predict();
    last_detection = cv::Rect_<float>(prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2), prediction.at<float>(3));
  }
}

cv::Rect Kalman::update(cv::Rect rect, bool hasDetected)
{
  float u, v, width, height;
  if (initialized)
  {
    cv::Mat measurement = cv::Mat::zeros(measureNum, 1, CV_32F); 
    if (hasDetected)
    {
      tracking_count = 0;
      float x = rect.x + (rect.width / 2);
      float y = rect.y + (rect.height / 2);
      measurement.at<float>(0) = (float)x;  
      measurement.at<float>(1) = (float)y; 
      measurement.at<float>(2) = (float)rect.width; 
      measurement.at<float>(3) = (float)rect.height; 
      cv::Mat estimated = KF.correct(measurement);  
      last_detection.x = rect.x;
      last_detection.y = rect.y;
      last_detection.width = rect.width;
      last_detection.height = rect.height;
      u = estimated.at<float>(0) - (estimated.at<float>(2) / 2);
      v = estimated.at<float>(1) - (estimated.at<float>(3) / 2);
      width = estimated.at<float>(2);
      height = estimated.at<float>(3);
    }
    else
    {
      tracking_count++;
      float x = last_detection.x + (last_detection.width / 2);
      float y = last_detection.y + (last_detection.height / 2);
      measurement.at<float>(0) = (float)x;  
      measurement.at<float>(1) = (float)y; 
      measurement.at<float>(2) = (float)last_detection.width; 
      measurement.at<float>(3) = (float)last_detection.height; 
      cv::Mat estimated = KF.correct(measurement);  
      last_detection.x = estimated.at<float>(0) - (estimated.at<float>(2) / 2);
      last_detection.y = estimated.at<float>(1) - (estimated.at<float>(3) / 2);
      last_detection.width = estimated.at<float>(2);
      last_detection.height = estimated.at<float>(3);  
      u = estimated.at<float>(0) - (estimated.at<float>(2) / 2);
      v = estimated.at<float>(1) - (estimated.at<float>(3) / 2);
      width = estimated.at<float>(2);
      height = estimated.at<float>(3);
    }
  }
  else
  {
    if(hasDetected)
    {
      init(rect);
    }
  }    
  isUpdated = true;
  return cv::Rect(u, v, width, height);
}



