#include "LightNet.h"

using namespace std;
using namespace Tn;
//using namespace Yolo;
using namespace cv;

vector<string> split(const string& str, char delim)
{
  stringstream ss(str);
  string token;
  vector<string> container;
  while (getline(ss, token, delim))
  {
    container.push_back(token);
  }

  return container;
}

float IOU_CALC(int box1[7], int box2[6])
{
	float minx1 = (float)box1[1];
	float maxx1 = (float)box1[3];
	float miny1 = (float)box1[2];
	float maxy1 = (float)box1[4];
	float minx2 = (float)box2[1];
	float maxx2 = (float)box2[3];
	float miny2 = (float)box2[2];
	float maxy2 = (float)box2[4];
	float distx = (((maxx2 - minx2) / 2) + minx2) - (((maxx1 - minx1) / 2) + minx1);
	float disty = (((maxy2 - miny2) / 2) + miny2) - (((maxy1 - miny1) / 2) + miny1);
	float dist = (disty * disty) + (distx * distx);

	float minxc, minyc, maxxc, maxyc;
	if (minx1 > minx2){minxc = minx2;}
	else{minxc = minx1;}
	if (miny1 > miny2){minyc = miny2;}
	else{minyc = miny1;}
	if (maxx1 > maxx2){maxxc = maxx1;}
	else{maxxc = maxx2;}
	if (maxy1 > maxy2){maxyc = maxy1;}
	else{maxyc = maxy2;}
	float normx = maxxc - minxc;
	float normy = maxyc - minyc;
	float norm = (normy * normy) + (normx * normx);

	if (minx1 > maxx2 || maxx1 < minx2 || miny1 > maxy2 || maxy1 < miny2 || box1[0] != box2[0])
	{
		return 0.0f;
	}
	else
	{
		float dx = std::min(maxx2, maxx1) - std::max(minx2, minx1);
		float dy = std::min(maxy2, maxy1) - std::max(miny2, miny1);
		float area1 = (maxx1 - minx1) * (maxy1 - miny1);
		float area2 = (maxx2 - minx2) * (maxy2 - miny2);
		float inter = dx * dy;
		float uni = area1 + area2 - inter; 
		float IoU = inter / uni;
		return (IoU + 1.1 * (dist / norm));
	}
}
void initi_all(const std::string& LightNet_TRT_model_path)
{
  string engineName_30deg_real(LightNet_TRT_model_path);
  engineName_30deg_real = engineName_30deg_real + "/resources/iclu30_v3.cfg";
  string engineName_60deg_real(LightNet_TRT_model_path);
  engineName_60deg_real = engineName_60deg_real + "/resources/iclu60_v3.cfg";

  string engineName_30deg_real1(LightNet_TRT_model_path);
  engineName_30deg_real1 = engineName_30deg_real1 + "/resources/yolov3_fp16_201208_30deg.engine";
  string engineName_60deg_real1(LightNet_TRT_model_path);
  engineName_60deg_real1 = engineName_60deg_real1 + "/resources/yolov3_fp16_201208_60deg.engine";

  config_v4_30.net_type = YOLOV4;
  config_v4_30.file_model_cfg = engineName_30deg_real;
  config_v4_30.file_model_weights = LightNet_TRT_model_path + "/resources/iclu30_v3.weights";
  // config_v4_30.calibration_image_list_file_txt = "../configs/calibration_images.txt";
  config_v4_30.inference_precison = INT8;
  config_v4_30.detect_thresh = 0.95;

  config_v4_60.net_type = YOLOV4;
  config_v4_60.file_model_cfg = engineName_60deg_real;
  config_v4_60.file_model_weights = LightNet_TRT_model_path + "/resources/iclu60_v3.weights";
  // config_v4_60.calibration_image_list_file_txt = "../configs/calibration_images.txt";
  config_v4_60.inference_precison = INT8;
  config_v4_60.detect_thresh = 0.95;

  detector_30->init(config_v4_60);
  detector_60->init(config_v4_60);

  for (int k = 0; k < 6; k++)
  {
    finLightStatus30[k] = false;
    preLightCount30[k] = 0;
    finLightStatus60[k] = false;
    preLightCount60[k] = 0;
  }
  for (int l = 0; l < 2; l++)
  {
    for (int k = 0; k < 6; k++)
    {
      finLightStatus[l][k] = false;
    }
    finLightDistance[l] = 0;
  }

  imageDimension.x = 608;
  imageDimension.y = 384;
  inputData.reserve(384 * 608 * 3 * 1);
  inputData_60deg.reserve(384 * 608 * 3 * 1);
  measurement = Mat::zeros(measureNum, 1, CV_32F);
  KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 10, 0, 0, 1, 0, 10, 0, 0, 1, 0, 0, 0, 0, 1);
  setIdentity(KF.measurementMatrix);
  setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
  setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
  setIdentity(KF.errorCovPost, Scalar::all(1));
}

void* preprocess_30deg(void* ptr)
{
  cv::Mat img_padding = cv::Mat(320, 608, CV_8UC3);
  inputData.clear();
  int top, bottom, left, right;
  top = 0;
  bottom = (342 - 320);
  left = 0;
  right = 0;
  copyMakeBorder(dImg, img_padding, top, bottom, left, right, BORDER_CONSTANT);

  inputData.push_back(img_padding);
}

void* preprocess_60deg(void* ptr)
{
  cv::Mat img_padding = cv::Mat(320, 608, CV_8UC3);

  int top, bottom, left, right;
  top = 0;
  bottom = (342 - 320);
  left = 0;
  right = 0;
  copyMakeBorder(dImg_60deg, img_padding, top, bottom, left, right, BORDER_CONSTANT);

  inputData_60deg.clear();
  inputData_60deg.push_back(img_padding);
}

void* postprocess_30deg(void* ptr)
{
  numOfDetection = numOfBulb30;
  filterBB(trafficLightBulb30, numOfDetection, _tlBulb30, numOfBulb30, 30);
  clusterBB(_tlBulb30, numOfBulb30, _tlBox30, numOfBox30);
  statusRecognize30(_tlBulb30, numOfBulb30, _tlBox30, numOfBox30, selectTL30, dImg, IsItTheFirstFrame, preLightCount30,
                    finLightStatus30);
#ifdef display_cv_window
  // Only for Visualization
  for (int k = 0; k < numOfBulb30; k++)
  {
    putText(dImg1, intToString(_tlBulb30[k][8]), Point(_tlBulb30[k][1], _tlBulb30[k][2]), 1, 1, Scalar(255, 255, 0), 2,
            8, 0);
    rectangle(dImg1, Point(_tlBulb30[k][1], _tlBulb30[k][2]), Point(_tlBulb30[k][3], _tlBulb30[k][4]),
              Scalar(255, 255, 0), 1, 8, 0);
  }
  if (selectTL30 > -1)
  {
    for (int k = 0; k < _tlBox30[selectTL30][0]; k++)
    {
      putText(dImg1, intToString(_tlBulb30[_tlBox30[selectTL30][k + 2]][8]),
              Point(_tlBulb30[_tlBox30[selectTL30][k + 2]][1], _tlBulb30[_tlBox30[selectTL30][k + 2]][2]), 1, 1,
              Scalar(0, 0, 255), 2, 8, 0);
      rectangle(dImg1, Point(_tlBulb30[_tlBox30[selectTL30][k + 2]][1], _tlBulb30[_tlBox30[selectTL30][k + 2]][2]),
                Point(_tlBulb30[_tlBox30[selectTL30][k + 2]][3], _tlBulb30[_tlBox30[selectTL30][k + 2]][4]),
                Scalar(0, 0, 255), 1, 8, 0);
    }
  }
#endif
}

void* postprocess_60deg(void* ptr)
{
  numOfDetection_60deg = numOfBulb60;
  filterBB(trafficLightBulb60, numOfDetection_60deg, _tlBulb60, numOfBulb60, 60);
  clusterBB(_tlBulb60, numOfBulb60, _tlBox60, numOfBox60);
  statusRecognize60(_tlBulb60, numOfBulb60, _tlBox60, numOfBox60, selectTL60, dImg_60deg, IsItTheFirstFrame,
                    preLightCount60, finLightStatus60);
#ifdef display_cv_window
  // Only for Visualization
  for (int k = 0; k < numOfBulb60; k++)
  {
    putText(dImg1_60deg, intToString(_tlBulb60[k][8]), Point(_tlBulb60[k][1], _tlBulb60[k][2]), 1, 1,
            Scalar(255, 255, 0), 2, 8, 0);
    rectangle(dImg1_60deg, Point(_tlBulb60[k][1], _tlBulb60[k][2]), Point(_tlBulb60[k][3], _tlBulb60[k][4]),
              Scalar(255, 255, 0), 1, 8, 0);
  }
  if (selectTL60 > -1)
  {
    for (int k = 0; k < _tlBox60[selectTL60][0]; k++)
    {
      putText(dImg1_60deg, intToString(_tlBulb60[_tlBox60[selectTL60][k + 2]][8]),
              Point(_tlBulb60[_tlBox60[selectTL60][k + 2]][1], _tlBulb60[_tlBox60[selectTL60][k + 2]][2]), 1, 1,
              Scalar(0, 0, 255), 2, 8, 0);
      rectangle(dImg1_60deg,
                Point(_tlBulb60[_tlBox60[selectTL60][k + 2]][1], _tlBulb60[_tlBox60[selectTL60][k + 2]][2]),
                Point(_tlBulb60[_tlBox60[selectTL60][k + 2]][3], _tlBulb60[_tlBox60[selectTL60][k + 2]][4]),
                Scalar(0, 0, 255), 1, 8, 0);
    }
  }
#endif
}
stringstream openName;
char datapath[500];
int j = 2794;
void DoNet(cv_bridge::CvImagePtr cv_ptr_30deg, cv_bridge::CvImagePtr cv_ptr_60deg, struct TL_status* TL_status_info,
           struct TL_color* TL_color_info)
{
  frame_count++;

  dImg = cv_ptr_30deg->image;
  dImg_60deg = cv_ptr_60deg->image;

#ifdef display_cv_window
  dImg.copyTo(dImg1);
  dImg_60deg.copyTo(dImg1_60deg);
#endif

  pthread_create(&thread_preprocess_30deg, NULL, preprocess_30deg, NULL);
  pthread_create(&thread_preprocess_60deg, NULL, preprocess_60deg, NULL);

  pthread_join(thread_preprocess_30deg, NULL);
  pthread_join(thread_preprocess_60deg, NULL);
  detector_30->detect(inputData, batch_res_30);
  detector_60->detect(inputData_60deg, batch_res_60);
	int bbCountInput, bbCountOutput;
	int bbInput[50][6];  //class-xstart-ystart-xstop-ystop-enable
	int bbOutput[50][5]; //class-xstart-ystart-xstop-ystop
	float confInput[50], confOutput[50];
	float iou;

  numOfBulb30 = 0;
  numOfBulb60 = 0;
  for (int i = 0; i < inputData.size(); ++i)
  {
    bbCountInput = 0;
    bbCountOutput = 0;
    for (const auto& r : batch_res_30[i])
    {
      // Data Normalization
      bbInput[bbCountInput][0] = r.id;
      bbInput[bbCountInput][1] = r.rect.x;
      bbInput[bbCountInput][2] = r.rect.y;
      bbInput[bbCountInput][3] = r.rect.width + r.rect.x;
      bbInput[bbCountInput][4] = r.rect.height + r.rect.y;
      bbInput[bbCountInput][5] = 1;
      confInput[bbCountInput] = r.prob;
      bbCountInput++;
    }

    // Perform Non-suppresing Maximum
    for (int l = 0; l < bbCountInput; l++)
    {
      if (bbInput[l][5] == 1)
      {
        for (int k = 0; k < bbCountInput; k++)
        {
          if (l != k)
          {
            if (bbInput[k][5] == 1)
            {
              iou = IOU_CALC(bbInput[l], bbInput[k]);
              if (iou > 0.2)
              {
                if (bbInput[l][0] == bbInput[k][0])
                {
                  if (confInput[l] > confInput[k])
                  {
                    bbInput[k][5] = 0;
                  }
                  else
                  {
                    bbInput[l][5] = 0;
                  }
                }
                else
                {
                  if (iou > 0.8)
                  {
                    if (confInput[l] > confInput[k])
                    {
                      bbInput[k][5] = 0;
                    }
                    else
                    {
                      bbInput[l][5] = 0;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    // Display Detection Result
    for (int l = 0; l < bbCountInput; l++)
    {
      if (bbInput[l][5] == 1)
      {
        for (int d = 0; d < 5; d++)
        {
          bbOutput[bbCountOutput][d] = bbInput[l][d];
        }
        confOutput[bbCountOutput] = confInput[l];

        if (bbOutput[bbCountOutput][0] < 6)
        {
          trafficLightBulb30[numOfBulb30][0] = 1;
          trafficLightBulb30[numOfBulb30][1] = bbOutput[bbCountOutput][1];
          trafficLightBulb30[numOfBulb30][2] = bbOutput[bbCountOutput][2];
          trafficLightBulb30[numOfBulb30][3] = bbOutput[bbCountOutput][3];
          trafficLightBulb30[numOfBulb30][4] = bbOutput[bbCountOutput][4];
          trafficLightBulb30[numOfBulb30][5] = (bbOutput[bbCountOutput][3] - bbOutput[bbCountOutput][1]) *
                                               (bbOutput[bbCountOutput][4] - bbOutput[bbCountOutput][2]);
          trafficLightBulb30[numOfBulb30][6] = bbOutput[bbCountOutput][0];
          trafficLightBulb30[numOfBulb30][7] = -1;
          numOfBulb30++;
        }
        bbCountOutput++;
      }
    }
  }

  for (int i = 0; i < inputData_60deg.size(); ++i)
  {
    bbCountInput = 0;
    bbCountOutput = 0;
    for (const auto& r : batch_res_60[i])
    {
      // Data Normalization
      bbInput[bbCountInput][0] = r.id;
      bbInput[bbCountInput][1] = r.rect.x;
      bbInput[bbCountInput][2] = r.rect.y;
      bbInput[bbCountInput][3] = r.rect.width + r.rect.x;
      bbInput[bbCountInput][4] = r.rect.height + r.rect.y;
      bbInput[bbCountInput][5] = 1;
      confInput[bbCountInput] = r.prob;
      bbCountInput++;
    }

    // Perform Non-suppresing Maximum
    for (int l = 0; l < bbCountInput; l++)
    {
      if (bbInput[l][5] == 1)
      {
        for (int k = 0; k < bbCountInput; k++)
        {
          if (l != k)
          {
            if (bbInput[k][5] == 1)
            {
              iou = IOU_CALC(bbInput[l], bbInput[k]);
              if (iou > 0.2)
              {
                if (bbInput[l][0] == bbInput[k][0])
                {
                  if (confInput[l] > confInput[k])
                  {
                    bbInput[k][5] = 0;
                  }
                  else
                  {
                    bbInput[l][5] = 0;
                  }
                }
                else
                {
                  if (iou > 0.8)
                  {
                    if (confInput[l] > confInput[k])
                    {
                      bbInput[k][5] = 0;
                    }
                    else
                    {
                      bbInput[l][5] = 0;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    // Display Detection Result
    for (int l = 0; l < bbCountInput; l++)
    {
      if (bbInput[l][5] == 1)
      {
        for (int d = 0; d < 5; d++)
        {
          bbOutput[bbCountOutput][d] = bbInput[l][d];
        }
        confOutput[bbCountOutput] = confInput[l];
        if (bbOutput[bbCountOutput][0] < 6)
        {
          trafficLightBulb60[numOfBulb60][0] = 1;
          trafficLightBulb60[numOfBulb60][1] = bbOutput[bbCountOutput][1];
          trafficLightBulb60[numOfBulb60][2] = bbOutput[bbCountOutput][2];
          trafficLightBulb60[numOfBulb60][3] = bbOutput[bbCountOutput][3];
          trafficLightBulb60[numOfBulb60][4] = bbOutput[bbCountOutput][4];
          trafficLightBulb60[numOfBulb60][5] = (bbOutput[bbCountOutput][3] - bbOutput[bbCountOutput][1]) *
                                               (bbOutput[bbCountOutput][4] - bbOutput[bbCountOutput][2]);
          trafficLightBulb60[numOfBulb60][6] = bbOutput[bbCountOutput][0];
          trafficLightBulb60[numOfBulb60][7] = -1;
          numOfBulb60++;
        }
        bbCountOutput++;
      }
    }
  }
 

#ifdef Tainan
  // myfile.close();
  pthread_create(&thread_postprocess_30deg, NULL, postprocess_30deg, NULL);
  pthread_join(thread_postprocess_30deg, NULL);
  pthread_create(&thread_postprocess_60deg, NULL, postprocess_60deg, NULL);

  pthread_join(thread_postprocess_60deg, NULL);
  // Finalize Detection Results
  finalizeTrafficLight(finLightStatus30, _tlBox30[selectTL30][1], finLightStatus60, _tlBox60[selectTL60][1],
                       finLightStatus, finLightDistance);

  // printf("F-Distance = %d\n", finLightDistance[1]);

#endif

  // memset(&TL_status_info,0,sizeof(TL_status_info));
  TL_status_info->left_green = 0;
  TL_status_info->ahead_green = 0;
  TL_status_info->right_green = 0;
  TL_status_info->ros_msg_total = 0;
  TL_status_info->distance_light = 0;
#ifdef Tainan

  if (finLightDistance[1] != -1 && finLightDistance[1] <= 40)
  {
    TL_status_info->distance_light = finLightDistance[1];

    if (finLightStatus[1][0])
      TL_color_info->color_light = 1;
    if (finLightStatus[1][1])
      TL_color_info->color_light = 2;
    if (finLightStatus[1][2])
      TL_color_info->color_light = 3;

    if (finLightStatus[1][0] == finLightStatus[1][1] == finLightStatus[1][2] == 0)
      TL_color_info->color_light = 0;

    TL_status_info->ahead_green = finLightStatus[1][3];
    TL_status_info->right_green = finLightStatus[1][4];
    TL_status_info->left_green = finLightStatus[1][5];
  }

#endif
  TL_status_info->ros_msg_total =
      TL_status_info->right_green << 2 | TL_status_info->ahead_green << 1 | TL_status_info->left_green;

  if (!numOfBulb30 && !numOfBulb60)
  {
    TL_color_info->color_light = 0;
    TL_status_info->ros_msg_total = 0;
  }
  if (!numOfBulb60 && !numOfBulb30)
  {
    detect_count++;
  }
  else
  {
    detect_count = 0;
  }
  if (finLightDistance[1] != -1 && finLightDistance[1] <= 40)
  {
    Mat prediction = KF.predict();


    measurement.at<float>(0) = (float)TL_status_info->distance_light;

    // cvKalmanCorrect(kalman, measurement);
    KF.correct(measurement);

    if (detect_count >= detect_continuous_frame)
    {
      TL_status_info->distance_light = 0;
      TL_color_info->color_light = 0;
      TL_status_info->ros_msg_total = 0;
    }
    else
      TL_status_info->distance_light = prediction.at<float>(0);
    // TL_status_info->distance_light = predict_pt.x;
  }
  // printf("F-Distance_kalman = %d\n", predict_pt.x);

  outputs.clear();
  outputs_60deg.clear();
  // animateFin = Mat::zeros(100, 1220, CV_8UC3);
  // animateTrafficLight(animateFin, finLightStatus[1], finLightDistance[1]);

  IsItTheFirstFrame = false;

#ifdef display_cv_window
  animateFin = Mat::zeros(100, 1220, CV_8UC3);
  animateTrafficLight(animateFin, finLightStatus[1], TL_status_info->distance_light);
  //     animate30 = Mat::zeros(100, 1220, CV_8UC3);
  //     animateTrafficLight(animate30, finLightStatus30, _tlBody30[selectTL30][6]);
  //     imshow("Result-30", animate30);
  //     animate60 = Mat::zeros(100, 1220, CV_8UC3);
  //     animateTrafficLight(animate60, finLightStatus60, _tlBody60[selectTL60][6]);
  //    // printf("60 deg = %d\n",_tlBody60[selectTL60][6]);
  //     imshow("Result-60", animate60);

  cv::imshow("view_60deg", dImg1_60deg);
  cv::imshow("view_30deg", dImg1);
  cv::imshow("Result-Final", animateFin);

  // cv::imshow("view", cv_ptr->image);
  cv::waitKey(1);
#endif
}
