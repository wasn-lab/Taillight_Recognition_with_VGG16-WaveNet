#include "LightNet.h"

using namespace std;
using namespace Tn;
using namespace Yolo;
using namespace cv;

vector<string> split(const string &str, char delim)
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

void initi_all()
{
    clahe_30deg = createCLAHE(); clahe_30deg->setClipLimit(1); clahe_30deg->setTilesGridSize(Size(5, 5));
    clahe_60deg = createCLAHE(); clahe_60deg->setClipLimit(1); clahe_60deg->setTilesGridSize(Size(5, 5));

    fstream file_30deg, file_60deg;
    file_30deg.open(engineName_30deg, ios::binary | ios::in);
    file_60deg.open(engineName_60deg, ios::binary | ios::in);

    if (!file_30deg.is_open())
    {
        cout << "read engine file" << engineName_30deg << " failed" << endl;
        printf("No find TRT Engine file!! Re-creat Engine file, please wait.\n");
        auto outputNames = split(OUTPUTS, ',');
        vector<vector<float>> calibData;
        //string saveName = "yolov3_" + mode + ".engine";
        net_30deg.reset(new trtNet(INPUT_PROTOTXT_30deg, INPUT_CAFFEMODEL_30deg, outputNames, calibData, run_mode_30deg, batchSize));
        cout << "save Engine..." << engineName_30deg << endl;
        net_30deg->saveEngine(engineName_30deg);
    }
    else
    {
        net_30deg.reset(new trtNet(engineName_30deg));
        assert(net_30deg->getBatchSize() == batchSize);
    }

    if (!file_60deg.is_open())
    {
        cout << "read engine file" << engineName_60deg << " failed" << endl;
        printf("No find TRT Engine file!! Re-creat Engine file, please wait.\n");
        auto outputNames = split(OUTPUTS, ',');
        vector<vector<float>> calibData;
        //string saveName = "yolov3_" + mode + ".engine";
        net_60deg.reset(new trtNet(INPUT_PROTOTXT_60deg, INPUT_CAFFEMODEL_60deg, outputNames, calibData, run_mode_60deg, batchSize));
        cout << "save Engine..." << engineName_60deg << endl;
        net_60deg->saveEngine(engineName_60deg);
    }
    else
    {
        net_60deg.reset(new trtNet(engineName_60deg));
        assert(net_60deg->getBatchSize() == batchSize);
    }

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

    outputCount = net_30deg->getOutputSize() / sizeof(float);
    inputData.reserve(384 * 608 * 3 * 1);
    inputData_60deg.reserve(384 * 608 * 3 * 1);

    //kalman = cvCreateKalman(stateNum, measureNum, 0);   //state(x,y,detaX,detaY)
    //measurement = cvCreateMat(measureNum, 1, CV_32FC1); //measurement(x,y)
    measurement = Mat::zeros(measureNum, 1, CV_32F);
    //memcpy(kalman->transition_matrix->data.fl, A, sizeof(A));
    //memcpy(KF.transitionMatrix.data.fl, A, sizeof(A));
    KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 10, 0,0, 1, 0, 10,0, 0, 1, 0,0, 0, 0, 1);

    //cvSetIdentity(kalman->measurement_matrix, cvRealScalar(1));
    //cvSetIdentity(kalman->process_noise_cov, cvRealScalar(1e-5));
    //cvSetIdentity(kalman->measurement_noise_cov, cvRealScalar(1e-1));
    //cvSetIdentity(kalman->error_cov_post, cvRealScalar(1));

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
    setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(KF.errorCovPost, Scalar::all(1));
}

void DoNms(vector<Detection> &detections, int classes, float nmsThresh)
{
    auto t_start = chrono::high_resolution_clock::now();

    vector<vector<Detection>> resClass;
    resClass.resize(classes);

    for (const auto &item : detections)
        resClass[item.classId].push_back(item);

    auto iouCompute = [](float *lbox, float *rbox) {
        float interBox[] = {
            max(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), //left
            min(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), //right
            max(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), //top
            min(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), //bottom
        };

        if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
        return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
    };

    vector<Detection> result;
    for (int i = 0; i < classes; ++i)
    {
        auto &dets = resClass[i];
        if (dets.size() == 0)
            continue;

        sort(dets.begin(), dets.end(), [=](const Detection &left, const Detection &right) {
            return left.prob > right.prob;
        });

        for (unsigned int m = 0; m < dets.size(); ++m)
        {
            auto &item = dets[m];
            result.push_back(item);
            for (unsigned int n = m + 1; n < dets.size(); ++n)
            {
               // printf("iouCompute(item.bbox, dets[n].bbox) = %d\n", iouCompute(item.bbox, dets[n].bbox));

                if (iouCompute(item.bbox, dets[n].bbox) > nmsThresh)
                {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }

    //swap(detections,result);
    detections = move(result);

    auto t_end = chrono::high_resolution_clock::now();
    float total = chrono::duration<float, milli>(t_end - t_start).count();
    //cout << "Time taken for nms is " << total << " ms." << endl;
}

vector<float> prepareImage(cv::Mat &img)
{
    using namespace cv;

    int c = 3;
    int h = 384;
    int w = 608; //net w

    float scale = min(float(w) / 608, float(h) / 384);
    auto scaleSize = cv::Size(608 * scale, 384 * scale);

    //printf("scaleSize.x = %d scaleSize.y = %d\n",scaleSize.width,scaleSize.height);
    cv::Mat rgb;
    cv::Mat resized;
    cv::cvtColor(img, resized, CV_BGR2RGB);

    //cv::resize(rgb, resized, scaleSize, 0, 0, INTER_CUBIC);

    // cv::Mat cropped(h, w, CV_8UC3, 127);
    // Rect rect((w - scaleSize.width) / 2, (h - scaleSize.height) / 2, scaleSize.width, scaleSize.height);
    // resized.copyTo(cropped(rect));

    cv::Mat img_float;
    if (c == 3)
        resized.convertTo(img_float, CV_32FC3, 1 / 255.0);
    else
        resized.convertTo(img_float, CV_32FC1, 1 / 255.0);

    //HWC TO CHW
    vector<Mat> input_channels(c);
    cv::split(img_float, input_channels);

    vector<float> result(h * w * c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i)
    {
        memcpy(data, input_channels[i].data, channelLength * sizeof(float));
        data += channelLength;
    }

    return result;
}

vector<Bbox> postProcessImg(vector<Detection> &detections, int classes)
{
    using namespace cv;

    int h = 384; //net h
    int w = 608; //net w

    //scale bbox to img
    int width = w;
    int height = h;
    float scale = min(float(w) / width, float(h) / height);
    float scaleSize[] = {width * scale, height * scale};

    //correct box
    for (auto &item : detections)
    {
        auto &bbox = item.bbox;
        bbox[0] = (bbox[0] * w - (w - scaleSize[0]) / 2.f) / scaleSize[0];
        bbox[1] = (bbox[1] * h - (h - scaleSize[1]) / 2.f) / scaleSize[1];
        bbox[2] /= scaleSize[0];
        bbox[3] /= scaleSize[1];
    }

    //nms
    float nmsThresh = 0.45;
    if (nmsThresh > 0)
        DoNms(detections, classes, nmsThresh);

    vector<Bbox> boxes;
    for (const auto &item : detections)
    {
        auto &b = item.bbox;
        Bbox bbox =
            {
                item.classId,                                  //classId
                max(int((b[0] - b[2] / 2.) * width), 0),       //left
                min(int((b[0] + b[2] / 2.) * width), width),   //right
                max(int((b[1] - b[3] / 2.) * height), 0),      //top
                min(int((b[1] + b[3] / 2.) * height), height), //bot
                item.prob,                                     //score
                item.SignBox};
        boxes.push_back(bbox);
    }

    return boxes;
}

void *preprocess_30deg(void *ptr)
{
    vector<float> curInput = prepareImage(dImg);
    //vector<float> curInput = prepareImage(dImg_hdr);
    inputData.insert(inputData.end(), curInput.begin(), curInput.end());
}

void *preprocess_60deg(void *ptr)
{
    vector<float> curInput1 = prepareImage(dImg_60deg);
    //vector<float> curInput1 = prepareImage(dImg_60deg_hdr);

    inputData_60deg.insert(inputData_60deg.end(), curInput1.begin(), curInput1.end());
}

void *postprocess_30deg(void *ptr)
{
    numOfDetection = numOfBulb30;
    filterBB(trafficLightBulb30, numOfDetection, _tlBulb30, numOfBulb30, 30);
    clusterBB(_tlBulb30, numOfBulb30, _tlBox30, numOfBox30);
    statusRecognize30(_tlBulb30, numOfBulb30, _tlBox30, numOfBox30, selectTL30, dImg, IsItTheFirstFrame, preLightCount30, finLightStatus30);

    // //Filter Bounding Boxes
    // filterTrafficLight(trafficLightBody30, numOfBody30, trafficLightBulb30, numOfBulb30, _tlBody30, _tlBulb30, selectTL30, imageDimension, 30);
#ifdef display_cv_window
    //Only for Visualization
    for (int k = 0; k < numOfBulb30; k++)
    {
        putText(dImg1, intToString(_tlBulb30[k][8]), Point(_tlBulb30[k][1], _tlBulb30[k][2]), 1, 1, Scalar(255, 255, 0), 2, 8, 0);
        rectangle(dImg1, Point(_tlBulb30[k][1], _tlBulb30[k][2]), Point(_tlBulb30[k][3], _tlBulb30[k][4]), Scalar(255, 255, 0), 1, 8, 0);
    }
    if (selectTL30 > -1)
    {
        for (int k = 0; k < _tlBox30[selectTL30][0]; k++)
        {
            putText(dImg1, intToString(_tlBulb30[_tlBox30[selectTL30][k + 2]][8]), Point(_tlBulb30[_tlBox30[selectTL30][k + 2]][1], _tlBulb30[_tlBox30[selectTL30][k + 2]][2]), 1, 1, Scalar(0, 0, 255), 2, 8, 0);
            rectangle(dImg1, Point(_tlBulb30[_tlBox30[selectTL30][k + 2]][1], _tlBulb30[_tlBox30[selectTL30][k + 2]][2]), Point(_tlBulb30[_tlBox30[selectTL30][k + 2]][3], _tlBulb30[_tlBox30[selectTL30][k + 2]][4]), Scalar(0, 0, 255), 1, 8, 0);
        }
    }
#endif
}

void *postprocess_60deg(void *ptr)
{
    numOfDetection_60deg = numOfBulb60;
    filterBB(trafficLightBulb60, numOfDetection_60deg, _tlBulb60, numOfBulb60, 60);
    clusterBB(_tlBulb60, numOfBulb60, _tlBox60, numOfBox60);
    statusRecognize60(_tlBulb60, numOfBulb60, _tlBox60, numOfBox60, selectTL60, dImg_60deg, IsItTheFirstFrame, preLightCount60, finLightStatus60);
#ifdef display_cv_window
    //Only for Visualization
    for (int k = 0; k < numOfBulb60; k++)
    {
        putText(dImg1_60deg, intToString(_tlBulb60[k][8]), Point(_tlBulb60[k][1], _tlBulb60[k][2]), 1, 1, Scalar(255, 255, 0), 2, 8, 0);
        rectangle(dImg1_60deg, Point(_tlBulb60[k][1], _tlBulb60[k][2]), Point(_tlBulb60[k][3], _tlBulb60[k][4]), Scalar(255, 255, 0), 1, 8, 0);
    }
    if (selectTL60 > -1)
    {
        for (int k = 0; k < _tlBox60[selectTL60][0]; k++)
        {
            putText(dImg1_60deg, intToString(_tlBulb60[_tlBox60[selectTL60][k + 2]][8]), Point(_tlBulb60[_tlBox60[selectTL60][k + 2]][1], _tlBulb60[_tlBox60[selectTL60][k + 2]][2]), 1, 1, Scalar(0, 0, 255), 2, 8, 0);
            rectangle(dImg1_60deg, Point(_tlBulb60[_tlBox60[selectTL60][k + 2]][1], _tlBulb60[_tlBox60[selectTL60][k + 2]][2]), Point(_tlBulb60[_tlBox60[selectTL60][k + 2]][3], _tlBulb60[_tlBox60[selectTL60][k + 2]][4]), Scalar(0, 0, 255), 1, 8, 0);
        }
    }
#endif
}
stringstream openName;
char datapath[500];
int j = 2794;
void DoNet(cv_bridge::CvImagePtr cv_ptr_30deg, cv_bridge::CvImagePtr cv_ptr_60deg, struct TL_status *TL_status_info, struct TL_color *TL_color_info)
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

    net_30deg->doInference(inputData.data(), outputData_30deg.get(), 1);
    net_60deg->doInference(inputData_60deg.data(), outputData_60deg.get(), 1);

    //Get Output
    auto output = outputData_30deg.get();
    auto outputSize = net_30deg->getOutputSize() / sizeof(float) / 1;

    //Get Output
    auto output_60deg = outputData_60deg.get();
    auto outputSize_60deg = net_60deg->getOutputSize() / sizeof(float) / 1;
    int detN = 0;
    int detN1 = 0;
    for (int i = 0; i < 1; ++i)
    {

        //process 30 deg camera result
        //first detect count
        int detCount = output[0];
        //later detect result
        vector<Detection> result;
        result.resize(detCount);
        memcpy(result.data(), &output[1], detCount * sizeof(Detection));

        auto boxes = postProcessImg(result, classNum);
        outputs.emplace_back(boxes);

        output += outputSize;

        //process 60 deg camera result
        int detCount_60deg = output_60deg[0];

        vector<Detection> result_60deg;
        result_60deg.resize(detCount_60deg);
        memcpy(result_60deg.data(), &output_60deg[1], detCount_60deg * sizeof(Detection));

        auto boxes_60deg = postProcessImg(result_60deg, classNum);
        outputs_60deg.emplace_back(boxes_60deg);

        output_60deg += outputSize_60deg;
    }
    inputData.clear();
    inputData_60deg.clear();
    auto bbox = *outputs.begin();
    auto bbox_60deg = *outputs_60deg.begin();

    numOfBulb30 = 0;
    numOfBulb60 = 0;
#ifdef Tainan
    // 30 deg camera result
    for (const auto &item : bbox)
    {
        normClass = item.classId;
        if (normClass < 6)
        {
            trafficLightBulb30[numOfBulb30][0] = 1;
            trafficLightBulb30[numOfBulb30][1] = item.left;
            trafficLightBulb30[numOfBulb30][2] = item.top;
            trafficLightBulb30[numOfBulb30][3] = item.right;
            trafficLightBulb30[numOfBulb30][4] = item.bot;
            trafficLightBulb30[numOfBulb30][5] = (item.right - item.left) * (item.bot - item.top);
            trafficLightBulb30[numOfBulb30][6] = normClass;
            trafficLightBulb30[numOfBulb30][7] = -1;
            numOfBulb30++;
        }
    }

    for (const auto &item : bbox_60deg)
    {
        normClass = item.classId;
        if (normClass < 6)
        {
            trafficLightBulb60[numOfBulb60][0] = 1;
            trafficLightBulb60[numOfBulb60][1] = item.left;
            trafficLightBulb60[numOfBulb60][2] = item.top;
            trafficLightBulb60[numOfBulb60][3] = item.right;
            trafficLightBulb60[numOfBulb60][4] = item.bot;
            trafficLightBulb60[numOfBulb60][5] = (item.right - item.left) * (item.bot - item.top);
            trafficLightBulb60[numOfBulb60][6] = normClass;
            trafficLightBulb60[numOfBulb60][7] = -1;
            numOfBulb60++;
        }
    }
   // myfile.close();
    pthread_create(&thread_postprocess_30deg, NULL, postprocess_30deg, NULL);
    pthread_join(thread_postprocess_30deg, NULL);
    pthread_create(&thread_postprocess_60deg, NULL, postprocess_60deg, NULL);

    pthread_join(thread_postprocess_60deg, NULL);
    //Finalize Detection Results
    finalizeTrafficLight(finLightStatus30, _tlBox30[selectTL30][1], finLightStatus60, _tlBox60[selectTL60][1], finLightStatus, finLightDistance);

    //printf("F-Distance = %d\n", finLightDistance[1]);

#endif

    //memset(&TL_status_info,0,sizeof(TL_status_info));
    TL_status_info->left_green = 0;
    TL_status_info->ahead_green = 0;
    TL_status_info->right_green = 0;
    TL_status_info->ros_msg_total = 0;
    TL_status_info->distance_light = 0;
#ifdef Tainan

    if (finLightDistance[1] != -1 && finLightDistance[1] <=40)
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
    TL_status_info->ros_msg_total = TL_status_info->right_green << 2 | TL_status_info->ahead_green << 1 | TL_status_info->left_green;

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
    if (finLightDistance[1] != -1&& finLightDistance[1] <=40)
    {
        //const CvMat *prediction = cv::KalmanFilter::predict(kalman, 0);
        Mat prediction = KF.predict();
        //CvPoint predict_pt = cvPoint((int)prediction.data.fl[0], (int)prediction.data.fl[1]);
        //measurement->data.fl[0] = (float)TL_status_info->distance_light;

        measurement.at<float>(0) = (float)TL_status_info->distance_light;

        //cvKalmanCorrect(kalman, measurement);
        KF.correct(measurement);

        if (detect_count >= detect_continuous_frame)
        {
            TL_status_info->distance_light = 0;
            TL_color_info->color_light = 0;
            TL_status_info->ros_msg_total = 0;
        }
        else
            TL_status_info->distance_light = prediction.at<float>(0);
        //TL_status_info->distance_light = predict_pt.x;
    }
    //printf("F-Distance_kalman = %d\n", predict_pt.x);
    
    outputs.clear();
    outputs_60deg.clear();
    //animateFin = Mat::zeros(100, 1220, CV_8UC3);
   // animateTrafficLight(animateFin, finLightStatus[1], finLightDistance[1]);

    IsItTheFirstFrame = false;

#ifdef display_cv_window
    animateFin = Mat::zeros(100, 1220, CV_8UC3);
    animateTrafficLight(animateFin, finLightStatus[1], prediction.at<float>(0));
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

    //cv::imshow("view", cv_ptr->image);
    cv::waitKey(1);
#endif
}