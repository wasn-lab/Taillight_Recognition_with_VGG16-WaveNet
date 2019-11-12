#include "ros/ros.h"
#include "std_msgs/Header.h"
#include <ros/package.h>

#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <boost/thread.hpp>
#include <pthread.h>
#include <thread>
#include <future>

#include "drivenet/drivenet_120_1_b1.h"

#include <msgs/DetectedObjectArray.h>

using namespace DriveNet;

pthread_t thrdYolo;
pthread_t thrdInterp;
pthread_t thrdDisplay;
void* run_yolo(void* );
void* run_interp(void* );
void* run_display(void* );

bool isInferStop;
bool isInferData;
bool isInferData_0;
bool isInferData_1;
bool isInferData_2;
bool isInferData_3;
bool isCompressed = false;
bool isCalibration = false;

pthread_mutex_t mtxInfer;
pthread_cond_t cndInfer;

std::string cam120_0_topicName;
std::string cam120_1_topicName;
std::string cam120_2_topicName;
std::string cam120_3_topicName;
ros::Subscriber cam120_0;
ros::Subscriber cam120_1;
ros::Subscriber cam120_2;
ros::Subscriber cam120_3;
ros::Publisher pub120_0;
ros::Publisher pub120_1;
ros::Publisher pub120_2;
ros::Publisher pub120_3;
image_transport::Publisher pubImg_120_0;
image_transport::Publisher pubImg_120_1;
image_transport::Publisher pubImg_120_2;
image_transport::Publisher pubImg_120_3;
sensor_msgs::ImagePtr imgMsg;
msgs::DetectedObjectArray doa120_0;
msgs::DetectedObjectArray doa120_1;
msgs::DetectedObjectArray doa120_2;
msgs::DetectedObjectArray doa120_3;

int rawimg_w = 1920;
int rawimg_h = 1208;
int img_w = 1920;
int img_h = 1208;
int img_size = img_w*img_h;
int rawimg_size = rawimg_w*rawimg_h;

cv::Mat mat120_0;
cv::Mat mat120_0_rect;
cv::Mat mat120_0_display;
std::vector<ITRI_Bbox> vBBX120_0;

cv::Mat mat120_1;
cv::Mat mat120_1_rect;
cv::Mat mat120_1_display;
std::vector<ITRI_Bbox> vBBX120_1;

cv::Mat mat120_2;
cv::Mat mat120_2_rect;
cv::Mat mat120_2_display;
std::vector<ITRI_Bbox> vBBX120_2;

cv::Mat mat120_3;
cv::Mat mat120_3_rect;
cv::Mat mat120_3_display;
std::vector<ITRI_Bbox> vBBX120_3;

std::vector<std::vector<ITRI_Bbox>* > vbbx_output;

cv::Mat cameraMatrix, distCoeffs;

std::vector<cv::Mat*> matSrcs;
std::vector<uint32_t> matOrder;
std::vector<uint32_t> matId;
std::vector<std_msgs::Header> headers;
std::vector<int> dist_rows;
std::vector<int> dist_cols;

// Prepare cv::Mat
void image_init()
{
    if (input_resize == 1)
    {
        img_w = 608; 
        img_h = 384;
    }
    img_size = img_w*img_h;

    if (display_flag)
    {
        mat120_0_display = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(0));
        mat120_1_display = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(0));
        mat120_2_display = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(0));
        mat120_3_display = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(0));
    }
}

void sync_inference(int camOrder, int camId, std_msgs::Header& header, cv::Mat *mat, std::vector<ITRI_Bbox>* vbbx, int dist_w, int dist_h){
    pthread_mutex_lock(&mtxInfer);

    bool isPushData = false;
    if (camOrder == 0 && !isInferData_0) {isInferData_0 = true; isPushData = true;}
    if (camOrder == 1 && !isInferData_1) {isInferData_1 = true; isPushData = true;}
    if (camOrder == 2 && !isInferData_2) {isInferData_2 = true; isPushData = true;}
    if (camOrder == 3 && !isInferData_3) {isInferData_3 = true; isPushData = true;}

    if (isPushData)
    {
        matSrcs.push_back(mat);
        matOrder.push_back(camOrder);
        matId.push_back(camId);
        vbbx_output.push_back(vbbx);
        headers.push_back(header);
        dist_cols.push_back(dist_w);
        dist_rows.push_back(dist_h);
        std::cout << "Subscribe " <<  camera::topics[cam_ids_[camOrder]] << " image." << std::endl;
    }

    if(matOrder.size() == 4) {
        isInferData = true;
        pthread_cond_signal(&cndInfer);
    }
    pthread_mutex_unlock(&mtxInfer);
    
    while(isInferData) usleep(5);
    
}

void callback_120_0(const sensor_msgs::Image::ConstPtr &msg){
    cv_bridge::CvImageConstPtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    mat120_0 = cv_ptr->image;
    
    std_msgs::Header h = msg->header;
    if(!isInferData_0) sync_inference(0, 5, h, &mat120_0, &vBBX120_0, 1920, 1208);
}

void callback_120_1(const sensor_msgs::Image::ConstPtr &msg){
    cv_bridge::CvImageConstPtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    mat120_1 = cv_ptr->image;
    // calibrationImage(mat120_1, mat120_1_rect, cameraMatrix, distCoeffs);

    std_msgs::Header h = msg->header;
    if(!isInferData_1)  sync_inference(1, 6, h, &mat120_1, &vBBX120_1, 1920, 1208);
}

void callback_120_2(const sensor_msgs::Image::ConstPtr &msg){
    cv_bridge::CvImageConstPtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    mat120_2 = cv_ptr->image;
    // calibrationImage(mat120_2, mat120_2_rect, cameraMatrix, distCoeffs);

    std_msgs::Header h = msg->header;
    if(!isInferData_2) sync_inference(2, 7, h, &mat120_2, &vBBX120_2, 1920, 1208);
}

void callback_120_3(const sensor_msgs::Image::ConstPtr &msg){
    cv_bridge::CvImageConstPtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    mat120_3 = cv_ptr->image;
    // calibrationImage(mat120_3, mat120_3_rect, cameraMatrix, distCoeffs);

    std_msgs::Header h = msg->header;
    if(!isInferData_3) sync_inference(3, 8, h, &mat120_3, &vBBX120_3, 1920, 1208);
}

void callback_120_0_decode(sensor_msgs::CompressedImage compressImg){
    cv::imdecode(cv::Mat(compressImg.data),1).copyTo(mat120_0);
    // calibrationImage(mat120_0, mat120_0_rect, cameraMatrix, distCoeffs);

    if(!isInferData_0) sync_inference(0, 5, compressImg.header, &mat120_0, &vBBX120_0, 1920, 1208);
}

void callback_120_1_decode(sensor_msgs::CompressedImage compressImg){
    cv::imdecode(cv::Mat(compressImg.data),1).copyTo(mat120_1);
    // calibrationImage(mat120_1, mat120_1_rect, cameraMatrix, distCoeffs);

    if(!isInferData_1)  sync_inference(1, 6, compressImg.header, &mat120_1, &vBBX120_1, 1920, 1208);
}

void callback_120_2_decode(sensor_msgs::CompressedImage compressImg){
    cv::imdecode(cv::Mat(compressImg.data),1).copyTo(mat120_2);
    // calibrationImage(mat120_2, mat120_2_rect, cameraMatrix, distCoeffs);

    if(!isInferData_2)  sync_inference(2, 7, compressImg.header, &mat120_2, &vBBX120_2, 1920, 1208);
}

void callback_120_3_decode(sensor_msgs::CompressedImage compressImg){
    cv::imdecode(cv::Mat(compressImg.data),1).copyTo(mat120_3);
    // calibrationImage(mat120_3, mat120_3_rect, cameraMatrix, distCoeffs);

    if(!isInferData_3)  sync_inference(3, 8, compressImg.header, &mat120_3, &vBBX120_3, 1920, 1208);
}

void image_publisher(cv::Mat image, std_msgs::Header header, int camOrder)
{
    imgMsg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();

    if(camOrder == 0)
	    pubImg_120_0.publish(imgMsg);
    else if (camOrder == 1)
 	    pubImg_120_1.publish(imgMsg);
    else if (camOrder == 2)
	    pubImg_120_2.publish(imgMsg);
    else if (camOrder == 3)
	    pubImg_120_3.publish(imgMsg);
}

/// roslaunch drivenet drivenet120.launch
int main(int argc, char **argv)
{
    ros::init(argc, argv, "drivenet_120_1_b1");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    isInferStop = false;
    isInferData = false;

    if (ros::param::get(ros::this_node::getName()+"/car_id", car_id));
	if (ros::param::get(ros::this_node::getName()+"/standard_fps", standard_FPS));
	if (ros::param::get(ros::this_node::getName()+"/display", display_flag));
	if (ros::param::get(ros::this_node::getName()+"/input_resize", input_resize));
	if (ros::param::get(ros::this_node::getName()+"/imgResult_publish", imgResult_publish));

    cam120_0_topicName = camera::topics[cam_ids_[0]];
    cam120_1_topicName = camera::topics[cam_ids_[1]];
    cam120_2_topicName = camera::topics[cam_ids_[2]];
    cam120_3_topicName = camera::topics[cam_ids_[3]];

    if(isCompressed){
        cam120_0 = nh.subscribe(cam120_0_topicName + std::string("/compressed"), 1, callback_120_0_decode);
        cam120_1 = nh.subscribe(cam120_1_topicName + std::string("/compressed"), 1, callback_120_1_decode);
        cam120_2 = nh.subscribe(cam120_2_topicName + std::string("/compressed"), 1, callback_120_2_decode);
        cam120_3 = nh.subscribe(cam120_3_topicName + std::string("/compressed"), 1, callback_120_3_decode);
    }
    else{
        cam120_0 = nh.subscribe(cam120_0_topicName, 1, callback_120_0);
        cam120_1 = nh.subscribe(cam120_1_topicName, 1, callback_120_1);
        cam120_2 = nh.subscribe(cam120_2_topicName, 1, callback_120_2);
        cam120_3 = nh.subscribe(cam120_3_topicName, 1, callback_120_3);        
    }


    if(imgResult_publish){
        pubImg_120_0 = it.advertise(cam120_0_topicName + std::string("/detect_image"), 1);
        pubImg_120_1 = it.advertise(cam120_1_topicName + std::string("/detect_image"), 1);
        pubImg_120_2 = it.advertise(cam120_2_topicName + std::string("/detect_image"), 1);
        pubImg_120_3 = it.advertise(cam120_3_topicName + std::string("/detect_image"), 1);
    }

    pub120_0 = nh.advertise<msgs::DetectedObjectArray>("/CamObjRightFront", 4);
    pub120_1 = nh.advertise<msgs::DetectedObjectArray>("/CamObjRightBack", 4);
    pub120_2 = nh.advertise<msgs::DetectedObjectArray>("/CamObjLeftFront", 4);   
    pub120_3 = nh.advertise<msgs::DetectedObjectArray>("/CamObjLeftBack", 4);

    pthread_mutex_init(&mtxInfer,NULL);
    pthread_cond_init(&cndInfer,NULL);

    int ret = pthread_create(&thrdYolo, NULL, &run_yolo, NULL);    
    if (standard_FPS == 1)
        int retInterp= pthread_create(&thrdInterp, NULL, &run_interp, NULL);
    if (display_flag == 1)
        int retDisplay = pthread_create(&thrdDisplay, NULL, &run_display, NULL); 

    std::string pkg_path = ros::package::getPath("drivenet");
    std::string cfg_file = "/b1_yolo_120_1.cfg";
    image_init();
    yoloApp.init_yolo(pkg_path, cfg_file);
    distEst.init(car_id);

    if(isCalibration){
        cv::String calibMatrix_filepath = pkg_path + "/config/sf3324.yml";
        std::cout << "calibMatrix_filepath: " << calibMatrix_filepath << std::endl; 
        loadCalibrationMatrix(calibMatrix_filepath, cameraMatrix, distCoeffs);
    }

    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();

    isInferStop = true;
    pthread_join(thrdYolo, NULL);
    if (standard_FPS == 1)
        pthread_join(thrdInterp, NULL);
    if (display_flag == 1)
        pthread_join(thrdDisplay, NULL);   

    yoloApp.delete_yolo_infer();
    ros::shutdown();

    return 0;
}

void* run_interp(void* ){
    std::cout << "run_interp start" << std::endl;
    ros::Rate r(30);
	while(ros::ok() && !isInferStop)
    {
		pub120_0.publish(doa120_0);
		pub120_1.publish(doa120_1);
		pub120_2.publish(doa120_2);
		pub120_3.publish(doa120_3);

        r.sleep();
	}
    std::cout << "run_interp close" << std::endl;
	pthread_exit(0);
}

msgs::DetectedObject run_dist(ITRI_Bbox box, int camOrder, int camId){
    msgs::DetectedObject detObj;
    msgs::BoxPoint boxPoint;
    msgs::CamInfo camInfo;
    double *diatance_pixel_array;

    bool BoxPass_flag = false;
    if (camOrder == 0){
        BoxPass_flag = false;
    }
    else if(camOrder == 1){
        // boundary for front top 120
        // cv::Point LeftLinePoint1(127, 272);
        // cv::Point LeftLinePoint2(-1422, 1207);
        // cv::Point RightLinePoint1(1904, 272);
        // cv::Point RightLinePoint2(3548, 1207);
        // BoxPass_flag = CheckBoxInArea(RightLinePoint1, RightLinePoint2, LeftLinePoint1, LeftLinePoint2, box.x1, box.y2, box.x2, box.y2);

        // if (box.y2 < 319) BoxPass_flag = false;
        BoxPass_flag = false;
    }
    else if(camOrder == 2){
        BoxPass_flag = false;
    }

    if (BoxPass_flag)
    {
        boxPoint = distEst.Get3dBBox(box.x1, box.y1, box.x2, box.y2, box.label, camId);
        detObj.bPoint = boxPoint;
    }

    camInfo.u = box.x1;
    camInfo.v = box.y1;
    camInfo.width = box.x2 - box.x1;
    camInfo.height = box.y2 - box.y1;
    camInfo.prob = box.prob;

    detObj.classId = translate_label(box.label);
    detObj.camInfo = camInfo;
    detObj.fusionSourceId = 0;

    return detObj;
}

void* run_yolo(void* ){
    std::cout << "run_inference start" << std::endl;
    std::vector<std_msgs::Header> headers_tmp;
    std::vector<std::vector<ITRI_Bbox>* > vbbx_output_tmp;
    std::vector<cv::Mat*> matSrcs_tmp;
    std::vector<uint32_t> matOrder_tmp;
    std::vector<uint32_t> matId_tmp;
    std::vector<int> dist_cols_tmp;    
    std::vector<int> dist_rows_tmp;

    cv::Mat M_display;
    cv::Mat M_display_tmp;
    std::vector<cv::Scalar> cls_color = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0) , cv::Scalar(125, 125, 125)};
    cv::Scalar class_color;

    ros::Rate r(30);
    while(ros::ok() && !isInferStop){

        pthread_mutex_lock(&mtxInfer);
        if(!isInferData) pthread_cond_wait(&cndInfer, &mtxInfer);
        pthread_mutex_unlock(&mtxInfer);

        headers_tmp = headers;
        vbbx_output_tmp = vbbx_output;
        matSrcs_tmp = matSrcs;
        matOrder_tmp = matOrder;
        matId_tmp = matId;
        dist_cols_tmp = dist_cols;
        dist_rows_tmp = dist_rows;

        // reset data
        headers.clear();
        matSrcs.clear();
        matOrder.clear();
        matId.clear();
        vBBX120_0.clear();
        vBBX120_1.clear();
        vBBX120_2.clear();
        vBBX120_3.clear();
        vbbx_output.clear();
        dist_cols.clear();
        dist_rows.clear();
        isInferData = false;
        isInferData_0 = false;
        isInferData_1 = false;
        isInferData_2 = false;
        isInferData_3 = false;

        if (!input_resize) yoloApp.input_preprocess(matSrcs_tmp); 
        else yoloApp.input_preprocess(matSrcs_tmp, matId_tmp, input_resize, dist_cols_tmp, dist_rows_tmp); 

        yoloApp.inference_yolo();
        yoloApp.get_yolo_result(&matOrder_tmp, vbbx_output_tmp);

        msgs::DetectedObjectArray doa;
        std::vector<msgs::DetectedObject> vDo;
        for(uint32_t ndx = 0; ndx < vbbx_output_tmp.size(); ndx++){
            std::vector<ITRI_Bbox>* tmpBBx = vbbx_output_tmp[ndx];
            if(imgResult_publish || display_flag)  
            {
                cv::resize((*matSrcs_tmp[ndx]), M_display_tmp, cv::Size(rawimg_w, rawimg_h), 0, 0, 0);
                M_display = M_display_tmp;
            }
            msgs::DetectedObject detObj;
            int cam_order = matOrder_tmp[ndx];
            int cam_id = matId_tmp[ndx];
            std::vector<std::future<msgs::DetectedObject>> pool;
            for(auto const& box: *tmpBBx) {
                if(translate_label(box.label) == 0) continue;
                pool.push_back(std::async(std::launch::async, run_dist, box, cam_order, cam_id));
                if(imgResult_publish || display_flag)
                {
                    class_color = get_labelColor(cls_color, box.label);
                    cv::rectangle(M_display, cvPoint(box.x1, box.y1), cvPoint(box.x2, box.y2), class_color, 8);
                }
            }
            for(int i = 0; i < pool.size(); i++)
            {
                detObj = pool[i].get();
                vDo.push_back(detObj);
                if(display_flag)
                {   
                    if(detObj.bPoint.p0.x != 0 && detObj.bPoint.p0.z != 0){
                        int distMeter_p0x, distMeter_p3x, distMeter_p0y, distMeter_p3y;
                        if (cam_order == 0)
                        {
                            distMeter_p0x = detObj.bPoint.p3.x;
                            distMeter_p3x = detObj.bPoint.p7.y;
                            distMeter_p0y = detObj.bPoint.p3.y;  
                            distMeter_p3y = detObj.bPoint.p7.y;
                        }                    
                        else if (cam_order == 2)
                        {
                            distMeter_p0x = detObj.bPoint.p4.x;
                            distMeter_p3x = detObj.bPoint.p0.y;
                            distMeter_p0y = detObj.bPoint.p4.y;  
                            distMeter_p3y = detObj.bPoint.p0.y;
                        }
                        else
                        {
                            distMeter_p0x = detObj.bPoint.p0.x;
                            distMeter_p3x = detObj.bPoint.p3.x;
                            distMeter_p0y = detObj.bPoint.p0.y;                    
                            distMeter_p3y = detObj.bPoint.p3.y;                        
                        }

                        int x1 = detObj.camInfo.u;
                        int x2 = detObj.camInfo.u + detObj.camInfo.width;
                        int y2 = detObj.camInfo.v + detObj.camInfo.height;

                        cv::putText(M_display, std::to_string(distMeter_p0x) + "," + std::to_string(distMeter_p0y), cvPoint(x1 - 100, y2 + 10), 0, 1, class_color, 2);
                        cv::putText(M_display, std::to_string(distMeter_p3x) + "," + std::to_string(distMeter_p3y), cvPoint(x2 + 10, y2 + 10), 0, 1, class_color, 2);
                    }
                }
            }
			doa.header = headers_tmp[ndx];
            doa.header.frame_id = "lidar";
            doa.objects = vDo;

            if(cam_order == 0){
                if (standard_FPS == 1) doa120_0 = doa;
                else pub120_0.publish(doa); 

                if(imgResult_publish || display_flag) {
                    mat120_0_display = M_display.clone();

                    if(imgResult_publish){
                        image_publisher(mat120_0_display, headers_tmp[ndx], 0);
                    }
                }
            }else if(cam_order == 1){
                if (standard_FPS == 1) doa120_1 = doa;
                else pub120_1.publish(doa);

                if(imgResult_publish || display_flag) {
                    mat120_1_display = M_display.clone();

                    if(imgResult_publish){
                        image_publisher(mat120_1_display, headers_tmp[ndx], 1);
                    }
                }
            }else if(cam_order == 2){
                if (standard_FPS == 1) doa120_2 = doa;
                else pub120_2.publish(doa);

                if(imgResult_publish || display_flag){
                    mat120_2_display = M_display.clone();

                    if(imgResult_publish){
                        image_publisher(mat120_2_display, headers_tmp[ndx], 2);
                    }
                }
            }else if(cam_order == 3){
                if (standard_FPS == 1) doa120_3 = doa;
                else pub120_3.publish(doa);
                                
                if(imgResult_publish || display_flag){
                    mat120_3_display = M_display.clone();

                    if(imgResult_publish){
                        image_publisher(mat120_3_display, headers_tmp[ndx], 3);
                    }
                }
            }
            vDo.clear();        
        }

        // reset data
        headers_tmp.clear();
        matSrcs_tmp.clear();
        matOrder_tmp.clear();
        matId_tmp.clear();
        vbbx_output_tmp.clear();
        dist_cols_tmp.clear();
        dist_rows_tmp.clear();
        r.sleep();
    }

    std::cout << "run_inference close" << std::endl;
    pthread_exit(0);
}

void* run_display(void* ){
    std::cout << "run_display start" << std::endl;
    cv::namedWindow("RightFront-120", CV_WINDOW_NORMAL);
    cv::namedWindow("RightBack-120", CV_WINDOW_NORMAL);   
    cv::namedWindow("LeftFront-120", CV_WINDOW_NORMAL);
    cv::namedWindow("LeftBack-120", CV_WINDOW_NORMAL);
    cv::resizeWindow("RightFront-120", 480, 360);
    cv::resizeWindow("RightBack-120", 480, 360);
    cv::resizeWindow("LeftFront-120", 480, 360);
    cv::resizeWindow("LeftBack-120", 480, 360);
    cv::moveWindow("RightFront-120", 1025, 360);
    cv::moveWindow("RightBack-120", 1500, 360);   
    cv::moveWindow("LeftFront-120", 545, 360);   
    cv::moveWindow("LeftBack-120", 0, 360);

    int marker_h = 0;
    marker_h = 590;  

    cv::Point BoundaryMarker1, BoundaryMarker2, BoundaryMarker3, BoundaryMarker4;
    BoundaryMarker1 = cv::Point(img_w/2 + 20, marker_h);
    BoundaryMarker2 = cv::Point(img_w/2 - 20, marker_h);
    BoundaryMarker3 = cv::Point(img_w/2, marker_h + 20);
    BoundaryMarker4 = cv::Point(img_w/2, marker_h - 20);

    ros::Rate r(10);
	while(ros::ok() && !isInferStop)
    {
        if (mat120_0_display.cols*mat120_0_display.rows == rawimg_size && mat120_1_display.cols*mat120_1_display.rows == rawimg_size 
            && mat120_2_display.cols*mat120_2_display.rows == rawimg_size
            && mat120_3_display.cols*mat120_3_display.rows == rawimg_size)
        {
            cv::line(mat120_1_display, BoundaryMarker1, BoundaryMarker2, cv::Scalar(255, 255, 255), 1);
            cv::line(mat120_1_display, BoundaryMarker3, BoundaryMarker4, cv::Scalar(255, 255, 255), 1);
            cv::imshow("RightFront-120", mat120_0_display);
            cv::imshow("RightBack-120", mat120_1_display);
            cv::imshow("LeftFront-120", mat120_2_display);
            cv::imshow("LeftBack-120", mat120_3_display);
            cv::waitKey(1);
        }
        r.sleep();
	}

    std::cout << "run_display close" << std::endl;
	pthread_exit(0);
}
