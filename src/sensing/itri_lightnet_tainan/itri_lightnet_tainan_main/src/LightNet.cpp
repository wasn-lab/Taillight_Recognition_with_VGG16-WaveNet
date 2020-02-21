#include "LightNet.h"

using namespace std;
using namespace Tn;
using namespace Yolo;
using namespace cv;

vector<string> split(const string& str, char delim)
{
    stringstream ss(str);
    string token;
    vector<string> container;
    while (getline(ss, token, delim)) {
        container.push_back(token);
    }

    return container;
}

void initi_all()
{
    fstream file;
    file.open(engineName, ios::binary | ios::in);
    if (!file.is_open())
    {
        cout << "read engine file" << engineName << " failed" << endl;
        printf("No find TRT Engine file!! Re-creat Engine file, please wait.\n");
        auto outputNames = split(OUTPUTS, ',');
        vector<vector<float>> calibData;
        //string saveName = "yolov3_" + mode + ".engine";
        net.reset(new trtNet(INPUT_PROTOTXT, INPUT_CAFFEMODEL, outputNames, calibData, run_mode, batchSize));
        cout << "save Engine..." << engineName << endl;
        net->saveEngine(engineName);
    }
    else
    {
        net.reset(new trtNet(engineName));
        assert(net->getBatchSize() == batchSize);
    }

    for (int k = 0; k < 6; k++)
	{
		finLightStatus[k] = false;
		preLightStatus[k] = false;
	}


    imageDimension.x = 608;
    imageDimension.y = 384;

    outputCount = net->getOutputSize() / sizeof(float);
    inputData.reserve(384 * 608 * 3 * 1);

    kalman = cvCreateKalman(stateNum, measureNum, 0);   //state(x,y,detaX,detaY)
    measurement = cvCreateMat(measureNum, 1, CV_32FC1); //measurement(x,y)
    memcpy(kalman->transition_matrix->data.fl, A, sizeof(A));
    cvSetIdentity(kalman->measurement_matrix, cvRealScalar(1));
    cvSetIdentity(kalman->process_noise_cov, cvRealScalar(1e-5));
    cvSetIdentity(kalman->measurement_noise_cov, cvRealScalar(1e-1));
    cvSetIdentity(kalman->error_cov_post, cvRealScalar(1));
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

    //printf("scaleSize.x = %d scaleSize.y\n",scaleSize.x,scaleSize.y);
    cv::Mat rgb;
    cv::cvtColor(img, rgb, CV_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, scaleSize, 0, 0, INTER_CUBIC);

    cv::Mat cropped(h, w, CV_8UC3, 127);
    Rect rect((w - scaleSize.width) / 2, (h - scaleSize.height) / 2, scaleSize.width, scaleSize.height);
    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    if (c == 3)
        cropped.convertTo(img_float, CV_32FC3, 1 / 255.0);
    else
        cropped.convertTo(img_float, CV_32FC1, 1 / 255.0);

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
    float nmsThresh = 0.1;
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
        /* if(item.SignBox>=0){
            SignBox_count += 1;
        }*/
        boxes.push_back(bbox);
    }

    return boxes;
}

void DoNet(cv_bridge::CvImagePtr cv_ptr, struct TL_status *TL_status_info, struct TL_color *TL_color_info)
{
    dImg = cv_ptr->image;
    dImg.copyTo(dImg1);
    vector<float> curInput = prepareImage(dImg);
    inputData.insert(inputData.end(), curInput.begin(), curInput.end());

    //memcpy(image.data,inputData.data(),inputData.size()*sizeof(float));

    net->doInference(inputData.data(), outputData.get(), 1);
    //Get Output
    auto output = outputData.get();
    auto outputSize = net->getOutputSize() / sizeof(float) / 1;
    int detN = 0;
    // SignBox_count = 0;
    for (int i = 0; i < 1; ++i)
    {
        //first detect count
        int detCount = output[0];
        //later detect result
        vector<Detection> result;
        result.resize(detCount);
        memcpy(result.data(), &output[1], detCount * sizeof(Detection));

        auto boxes = postProcessImg(result, classNum);
        outputs.emplace_back(boxes);
        //detN = boxes.size() + SignBox_count;

        output += outputSize;
        //std::cout << detCount << " outSize" << std::endl;
    }
    inputData.clear();
    auto bbox = *outputs.begin();
    numOfBulb = 0;
    numOfBody = 0;
    numOfDetection = 0;
    for (const auto &item : bbox)
    {
#ifdef ITRI_Field
        if (item.classId == 0)
        {
            trafficLightBody[numOfBody][0] = 1;
            trafficLightBody[numOfBody][1] = item.left;
            trafficLightBody[numOfBody][2] = item.top;
            trafficLightBody[numOfBody][3] = item.right;
            trafficLightBody[numOfBody][4] = item.bot;
            trafficLightBody[numOfBody][5] = (item.right - item.left) * (item.bot - item.top);
            numOfBody++;
        }
        else
        {
            trafficLightBulb[numOfBulb][0] = 1;
            trafficLightBulb[numOfBulb][1] = item.left;
            trafficLightBulb[numOfBulb][2] = item.top;
            trafficLightBulb[numOfBulb][3] = item.right;
            trafficLightBulb[numOfBulb][4] = item.bot;
            trafficLightBulb[numOfBulb][5] = (item.right - item.left) * (item.bot - item.top);
            trafficLightBulb[numOfBulb][6] = item.classId;
            trafficLightBulb[numOfBulb][7] = -1;
            numOfBulb++;
        }
#endif
#ifdef Tainan
    if((item.classId >=74 && item.classId <=79) || (item.classId ==90) )
    {
        normClass = item.classId - 74;
        if (normClass == 16) 
        {
            trafficLightBody[numOfBody][0] = 1;
            trafficLightBody[numOfBody][1] = item.left;
            trafficLightBody[numOfBody][2] = item.top;
            trafficLightBody[numOfBody][3] = item.right;
            trafficLightBody[numOfBody][4] = item.bot;
            trafficLightBody[numOfBody][5] = (item.right - item.left) * (item.bot - item.top);
            numOfBody++;
        }
        else
        {
            trafficLightBulb[numOfBulb][0] = 1;
            trafficLightBulb[numOfBulb][1] = item.left;
            trafficLightBulb[numOfBulb][2] = item.top;
            trafficLightBulb[numOfBulb][3] = item.right;
            trafficLightBulb[numOfBulb][4] = item.bot;
            trafficLightBulb[numOfBulb][5] = (item.right - item.left) * (item.bot - item.top);
            trafficLightBulb[numOfBulb][6] = normClass;
            trafficLightBulb[numOfBulb][7] = -1;
            numOfBulb++;
        }
       // cout << "class=" << item.classId << " prob=" << item.score * 100 << endl;
    }
#endif
/*
#ifdef display_cv_window
        cv::rectangle(dImg, cv::Point(item.left, item.top), cv::Point(item.right, item.bot), cv::Scalar(0, 0, 255), 1, 8, 0);
        if (item.SignBox >= 0)
        {
            cv::putText(dImg, class_name[item.classId] + ',' + class_name[item.SignBox], cv::Point(item.left, item.top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1, 2);
        }
        else
        {
            cv::putText(dImg, class_name[item.classId], cv::Point(item.left, item.top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1, 2);
        }
#endif
*/
        
        //cout << "left=" << item.left << " right=" << item.right << " top=" << item.top << " bot=" << item.bot << endl;

        //myfile << item.left << " " << item.top << " " << item.right << " " << item.bot << "\n";
    }
    //printf("numOfBulb =%d\n", numOfBulb);
#ifdef Tainan
    filterTrafficLight(trafficLightBody, numOfBody, trafficLightBulb, numOfBulb, _tlBody, _tlBulb, selectTL, imageDimension);
    if (selectTL > -1)
    {
        putText(dImg, intToString(selectTL), Point(_tlBody[selectTL][1], _tlBody[selectTL][2]), 1, 1, Scalar(0, 120, 255), 2, 8, 0);
        rectangle(dImg, Point(_tlBody[selectTL][1], _tlBody[selectTL][2]), Point(_tlBody[selectTL][3], _tlBody[selectTL][4]), Scalar(0, 120, 255), 1, 8, 0);

        //Improve Dimensionality of Bounding Box
        improveTrafficLight(_tlBody[selectTL], selectTL, insideBulb, numOfBulb, _tlBulb, dImg1);

        //Improve Detectability of Bounding Box
        if (_tlBody[selectTL][0] == 0)
        {
            //Propose Bounding Box
            proposeTrafficLight(_tlBody[selectTL], selectTL, insideBulb, numOfBulb, _tlBulb, dImg1);
        }
        else
        {
            //Verify Bounding Box
            verifyTrafficLight(_tlBody[selectTL], insideBulb, _tlBulb, dImg1);

            //Find Miss Bounding Box
            findTrafficLight(_tlBody[selectTL], selectTL, insideBulb, numOfBulb, _tlBulb, dImg1);
        }

        //Visualize Bounding Box
        for (int k = 0; k < _tlBody[selectTL][8]; k++)
        {
            rectangle(dImg, Point(_tlBulb[insideBulb[k]][1], _tlBulb[insideBulb[k]][2]), Point(_tlBulb[insideBulb[k]][3], _tlBulb[insideBulb[k]][4]), Scalar(255, 255, 0), 1, 8, 0);
        }
    }
    finalizeTrafficLight(selectTL, _tlBody, insideBulb, _tlBulb, finLightStatus, preLightStatus);

#ifdef printf_debug
    cout << "F-RED " << finLightStatus[0] << "  ";
    cout << "F-YLW " << finLightStatus[1] << "  ";
    cout << "F-GRE " << finLightStatus[2] << "  ";
    cout << "A-AHD " << finLightStatus[3] << "  ";
    cout << "A-RGT " << finLightStatus[4] << "  ";
    cout << "A-LFT " << finLightStatus[5] << "  ";
#endif
    if (selectTL > -1)
    {
        TL_status_info->distance_light = _tlBody[selectTL][6];
#ifdef printf_debug
        cout << "DIST  " << _tlBody[selectTL][6] << "meter";
#endif
    }
#ifdef printf_debug
    cout << endl;
#endif

#endif
#ifdef ITRI_Field
    filterTrafficLight(trafficLightBody, numOfBody, trafficLightBulb, numOfBulb, _tlBody, _tlBulb, imageDimension);
    proposeTrafficLight(_tlBody, numOfBody, dImg, _tlBulb, numOfBulb);
#endif    

   
    //memset(&TL_status_info,0,sizeof(TL_status_info));
    TL_status_info->left_green = 0;
    TL_status_info->ahead_green = 0;
    TL_status_info->right_green = 0;
    TL_status_info->ros_msg_total = 0;
#ifdef ITRI_Field
    for (int k = 0; k < numOfBulb; k++)
    {
        string netResult, predictColor;
        if (_tlBulb[k][7] == -1)
        {
            predictColor = "UNKNOWN";
            TL_color_info->color_light = 0;
        }
        else if (_tlBulb[k][7] == 0)
        {
            predictColor = "RED";
            TL_color_info->color_light = 1;
        }
        else if (_tlBulb[k][7] == 1)
        {
            predictColor = "YELLOW";
            TL_color_info->color_light = 2;
        }
        else if (_tlBulb[k][7] == 2)
        {
            predictColor = "LEFT GREEN";
            TL_status_info->left_green = 1;
        }
        else if (_tlBulb[k][7] == 3)
        {
            predictColor = "AHEAD GREEN";
            TL_status_info->ahead_green = 1;
        }
        else if (_tlBulb[k][7] == 4)
        {
            predictColor = "RIGHT GREEN";
            TL_status_info->right_green = 1;
        }
        //cout << "ADDED AS = " << predictColor << " DISTANCE = " << _tlBody[_tlBulb[k][8]][6] << endl;

        if (_tlBody[_tlBulb[k][8]][6] != 0)
            TL_status_info->distance_light = _tlBody[_tlBulb[k][8]][6];
#ifdef display_cv_window
        cv::putText(dImg, predictColor, Point(_tlBulb[k][1], _tlBulb[k][2]), 1, 1, Scalar(255, 255, 0), 2, 8, 0);
        cv::rectangle(dImg, Point(_tlBulb[k][1], _tlBulb[k][2]), Point(_tlBulb[k][3], _tlBulb[k][4]), Scalar(255, 255, 0), 1, 8, 0);
#endif
    }
#endif

#ifdef Tainan
    if (finLightStatus[0])
        TL_color_info->color_light = 1;
    if (finLightStatus[1])
        TL_color_info->color_light = 2;
    if (finLightStatus[2])
        TL_color_info->color_light = 3;

        if(finLightStatus[0] == finLightStatus[1] == finLightStatus[2] ==0)
        TL_color_info->color_light = 0;

    //TL_color_info->color_light = finLightStatus[0]; //red
    //TL_color_info->color_light = finLightStatus[1]; //yello
    //TL_color_info->color_light = finLightStatus[2]; //green

    TL_status_info->ahead_green = finLightStatus[3];
    TL_status_info->right_green = finLightStatus[4];
    TL_status_info->left_green = finLightStatus[5];
#endif
    TL_status_info->ros_msg_total = TL_status_info->right_green << 2 | TL_status_info->ahead_green << 1 | TL_status_info->left_green;

    if (!numOfBody)
    {
        TL_color_info->color_light = 0;
        TL_status_info->ros_msg_total = 0;
    }
    if (!numOfBulb)
    {
        detect_count++;
    }
    else
    {
        detect_count = 0;
    }

    const CvMat *prediction = cvKalmanPredict(kalman, 0);
    CvPoint predict_pt = cvPoint((int)prediction->data.fl[0], (int)prediction->data.fl[1]);
    measurement->data.fl[0] = (float)TL_status_info->distance_light;
    cvKalmanCorrect(kalman, measurement);

    if (detect_count >= detect_continuous_frame)
    {
        TL_status_info->distance_light = 0;
        TL_color_info->color_light = 0;
        TL_status_info->ros_msg_total = 0;
    }
    else
        TL_status_info->distance_light = predict_pt.x;

    outputs.clear();
#ifdef display_cv_window
    cv::imshow("view", dImg);
    //cv::imshow("view", cv_ptr->image);
    cv::waitKey(1);
#endif
}