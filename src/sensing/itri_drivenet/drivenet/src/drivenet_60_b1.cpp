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

#include "drivenet/trt_yolo_interface.h"
#include <msgs/DetectedObjectArray.h>
#include "drivenet/DistanceEstimation_b1.h"

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
bool iscompressed = false;

/// launch param
int car_id = 1;
bool standard_FPS = 0;
bool display_flag = 0;
bool input_resize = 0; //grabber input mode 0: 1920x1208, 1:608x384 yolo format
bool imgResult_publish = 0; 

pthread_mutex_t mtxInfer;
pthread_cond_t cndInfer;

std::string cam60_0_topicName;
std::string cam60_1_topicName;
std::string cam60_2_topicName;
ros::Subscriber cam60_0;
ros::Subscriber cam60_1;
ros::Subscriber cam60_2;
ros::Publisher pub60_0;
ros::Publisher pub60_1;
ros::Publisher pub60_2;
image_transport::Publisher pubImg_60_0;
image_transport::Publisher pubImg_60_1;
image_transport::Publisher pubImg_60_2;
sensor_msgs::ImagePtr imgMsg;
msgs::DetectedObjectArray doa60_0;
msgs::DetectedObjectArray doa60_1;
msgs::DetectedObjectArray doa60_2;

int img_w = 1920;
int img_h = 1208;
int rawimg_w = 1920;
int rawimg_h = 1208;
int img_size = img_w*img_h;
int rawimg_size = rawimg_w*rawimg_h;
cv::Mat mat60_0;
cv::Mat mat60_0_display;
std::vector<ITRI_Bbox> vBBX60_0;

cv::Mat mat60_1;
cv::Mat mat60_1_display;
std::vector<ITRI_Bbox> vBBX60_1;

cv::Mat mat60_2;
cv::Mat mat60_2_display;
std::vector<ITRI_Bbox> vBBX60_2;

std::vector<std::vector<ITRI_Bbox>* > vbbx_output;

std::vector<cv::Mat*> matSrcs;
std::vector<uint32_t> matOrder;
std::vector<uint32_t> matId;
std::vector<std_msgs::Header> headers;

DistanceEstimation de;
Yolo_app yoloApp;


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
        mat60_0_display = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(0));
        mat60_1_display = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(0));
        mat60_2_display = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(0));
    }
}

void sync_inference(int camOrder, int camId, std_msgs::Header& header, cv::Mat *mat, std::vector<ITRI_Bbox>* vbbx){

    pthread_mutex_lock(&mtxInfer);

    bool isPushData = false;
    if (camOrder == 0 && !isInferData_0) {isInferData_0 = true; isPushData = true;}
    if (camOrder == 1 && !isInferData_1) {isInferData_1 = true; isPushData = true;}
    if (camOrder == 2 && !isInferData_2) {isInferData_2 = true; isPushData = true;}

    if (isPushData)
    {
        matSrcs.push_back(mat);
        matOrder.push_back(camOrder);
        matId.push_back(camId);
        vbbx_output.push_back(vbbx);
        headers.push_back(header);
        std::cout << __FILE__ << __LINE__ << ", camOrder: " << camOrder << std::endl;
    }

    if(matOrder.size() == 3) {
        // std::cout << "collect 3 camera finish " << std::endl;
        isInferData = true;
        pthread_cond_signal(&cndInfer);
    }
    pthread_mutex_unlock(&mtxInfer);
    
    while(isInferData) usleep(5);
}

void callback_60_0(const sensor_msgs::Image::ConstPtr &msg){
    cv_bridge::CvImageConstPtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    mat60_0 = cv_ptr->image;

    std_msgs::Header h = msg->header;
    if(!isInferData_0) sync_inference(0, 1, h, &mat60_0, &vBBX60_0);
}

void callback_60_1(const sensor_msgs::Image::ConstPtr &msg){
    cv_bridge::CvImageConstPtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    mat60_1 = cv_ptr->image;

    std_msgs::Header h = msg->header;
    if(!isInferData_1)  sync_inference(1, 2, h, &mat60_1, &vBBX60_1);
}

void callback_60_2(const sensor_msgs::Image::ConstPtr &msg){
    cv_bridge::CvImageConstPtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    mat60_2 = cv_ptr->image;
    
    std_msgs::Header h = msg->header;
    if(!isInferData_2) sync_inference(2, 3, h, &mat60_2, &vBBX60_2);
}

void callback_60_0_decode(sensor_msgs::CompressedImage compressImg){
    cv::imdecode(cv::Mat(compressImg.data),1).copyTo(mat60_0);

    if(!isInferData_0) sync_inference(0, 1, compressImg.header, &mat60_0, &vBBX60_0);
}

void callback_60_1_decode(sensor_msgs::CompressedImage compressImg){
    cv::imdecode(cv::Mat(compressImg.data),1).copyTo(mat60_1);

    if(!isInferData_1)  sync_inference(1, 2, compressImg.header, &mat60_1, &vBBX60_1);
}

void callback_60_2_decode(sensor_msgs::CompressedImage compressImg){
    cv::imdecode(cv::Mat(compressImg.data),1).copyTo(mat60_2);
    
    if(!isInferData_2) sync_inference(2, 3, compressImg.header, &mat60_2, &vBBX60_2);
}

void image_publisher(cv::Mat image, std_msgs::Header header, int camOrder)
{
    imgMsg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();

    if(camOrder == 0)
	    pubImg_60_0.publish(imgMsg);
    else if (camOrder == 1)
 	    pubImg_60_1.publish(imgMsg);
    else if (camOrder == 2)
	    pubImg_60_2.publish(imgMsg);
}


/// roslaunch drivenet drivenet60.launch
int main(int argc, char **argv)
{
    ros::init(argc, argv, "drivenet_60_b1");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    isInferStop = false;
    isInferData = false;
    
    if (ros::param::get(ros::this_node::getName()+"/car_id", car_id));
	if (ros::param::get(ros::this_node::getName()+"/standard_fps", standard_FPS));
	if (ros::param::get(ros::this_node::getName()+"/display", display_flag));
	if (ros::param::get(ros::this_node::getName()+"/input_resize", input_resize));
	if (ros::param::get(ros::this_node::getName()+"/imgResult_publish", imgResult_publish));

    cam60_0_topicName = "gmsl_camera/0";
    cam60_1_topicName = "gmsl_camera/1";
    cam60_2_topicName = "gmsl_camera/2";
    
    if (iscompressed){
        cam60_0 = nh.subscribe(cam60_0_topicName + std::string("/compressed"), 1, callback_60_0_decode);
        cam60_1 = nh.subscribe(cam60_1_topicName + std::string("/compressed"), 1, callback_60_1_decode);
        cam60_2 = nh.subscribe(cam60_2_topicName + std::string("/compressed"), 1, callback_60_2_decode);
    }
    else{
        cam60_0 = nh.subscribe(cam60_0_topicName, 1, callback_60_0);
        cam60_1 = nh.subscribe(cam60_1_topicName, 1, callback_60_1);
        cam60_2 = nh.subscribe(cam60_2_topicName, 1, callback_60_2);
    }
    
    if(imgResult_publish){
        pubImg_60_0 = it.advertise(cam60_0_topicName + std::string("/detect_image"), 1);
        pubImg_60_1 = it.advertise(cam60_1_topicName + std::string("/detect_image"), 1);
        pubImg_60_2 = it.advertise(cam60_2_topicName + std::string("/detect_image"), 1);
    }

    pub60_0 = nh.advertise<msgs::DetectedObjectArray>("/CamObjFrontRight", 8);
    pub60_1 = nh.advertise<msgs::DetectedObjectArray>("/CamObjFrontCenter", 8);
    pub60_2 = nh.advertise<msgs::DetectedObjectArray>("/CamObjFrontLeft", 8);

    pthread_mutex_init(&mtxInfer,NULL);
    pthread_cond_init(&cndInfer,NULL);

    int ret = pthread_create(&thrdYolo, NULL, &run_yolo, NULL);    
    if (standard_FPS == 1)
        int retInterp = pthread_create(&thrdInterp, NULL, &run_interp, NULL);
    if (display_flag == 1)
        int retDisplay = pthread_create(&thrdDisplay, NULL, &run_display, NULL); 

    std::string pkg_path = ros::package::getPath("drivenet");
    std::string cfg_file = "/b1_yolo_60.cfg";
    image_init();
    yoloApp.init_yolo(pkg_path, cfg_file);
    de.init(car_id);

std::cout << __FILE__ << __LINE__ << std::endl;    
    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
std::cout << __FILE__ << __LINE__ << std::endl;

    isInferStop = true;
    pthread_join(thrdYolo, NULL);
    if (standard_FPS == 1)
        pthread_join(thrdInterp, NULL);
    if (display_flag == 1)
        pthread_join(thrdDisplay, NULL);   

std::cout << __FILE__ << __LINE__ << std::endl;

    yoloApp.delete_yolo_infer();
    ros::shutdown();

    return 0;
}


int translate_label(int label) {
    if(label == 0){
        return 1;
    }else if(label == 1){
        return 2;        
    }else if(label == 2){
        return 4;
    }else if(label == 3){
        return 3;
    }else if(label == 5){
        return 5;
    }else if(label == 7){
        return 6;
    }else {
        return 0;
    }
}
cv::Scalar get_labelColor(std::vector<cv::Scalar> colors, int label_id)
{
    cv::Scalar class_color;
    if(label_id == 0)
        class_color = colors[0];
    else if (label_id == 1 || label_id == 3)
        class_color = colors[1];
    else if (label_id == 2 || label_id == 5 || label_id == 7)
        class_color = colors[2];
    else
        class_color = colors[3];
    return class_color;
}
bool CheckBoxInArea(cv::Point RightLinePoint1, cv::Point RightLinePoint2, cv::Point LeftLinePoint1, cv::Point LeftLinePoint2, int object_x1, int object_y1, int object_x2, int object_y2)
{
    // printf("x1: %d, y1: %d, x2: %d, y2:%d\n", object_x1, object_y1, object_x2, object_y2);
    ///right
    int C1 = (RightLinePoint1.x - RightLinePoint2.x)*(object_y2 - RightLinePoint2.y) - (object_x2 - RightLinePoint2.x)*(RightLinePoint1.y - RightLinePoint2.y);
    int C2 = (RightLinePoint1.x - RightLinePoint2.x)*(object_y2 - RightLinePoint2.y) - (object_x1 - RightLinePoint2.x)*(RightLinePoint1.y - RightLinePoint2.y);
    ///left
    int C3 = (LeftLinePoint1.x - LeftLinePoint2.x)*(object_y2 - LeftLinePoint2.y) - (object_x2 - LeftLinePoint2.x)*(LeftLinePoint1.y - LeftLinePoint2.y);
    int C4 = (LeftLinePoint1.x - LeftLinePoint2.x)*(object_y2 - LeftLinePoint2.y) - (object_x1 - LeftLinePoint2.x)*(LeftLinePoint1.y - LeftLinePoint2.y);
    ///up
    int C5 = (RightLinePoint1.x - LeftLinePoint1.x)*(object_y2 - LeftLinePoint1.y) - (object_x2 - LeftLinePoint1.x)*(RightLinePoint1.y - LeftLinePoint1.y);
    int C6 = (RightLinePoint1.x - LeftLinePoint1.x)*(object_y2 - LeftLinePoint1.y) - (object_x1 - LeftLinePoint1.x)*(RightLinePoint1.y - LeftLinePoint1.y);
    ///bottom
    int C7 = (RightLinePoint2.x - LeftLinePoint2.x)*(object_y2 - LeftLinePoint2.y) - (object_x2 - LeftLinePoint2.x)*(RightLinePoint2.y - LeftLinePoint2.y);
    int C8 = (RightLinePoint2.x - LeftLinePoint2.x)*(object_y2 - LeftLinePoint2.y) - (object_x1 - LeftLinePoint2.x)*(RightLinePoint2.y - LeftLinePoint2.y);

    //printf("C1:%d, C3:%d, C5:%d, C7:%d\n", C1, C3, C5, C7);
    //printf("C2:%d, C4:%d, C6:%d, C8:%d\n", C2, C4, C6, C8);

    if (C1 < 0 && C3 > 0 && C5 > 0 && C7 < 0 && C2 < 0 && C4 > 0 && C6 > 0 && C8 < 0)
        return true;         
    else
        return false;
}

void* run_interp(void* ){
std::cout << "run_interp start" << std::endl;
    ros::Rate r(30);    
	while(ros::ok() && !isInferStop)
    {
		pub60_0.publish(doa60_0);
		pub60_1.publish(doa60_1);
		pub60_2.publish(doa60_2);
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
    else if(camOrder == 1)
    {
        cv::Point LeftLinePoint1(611, 945);
        cv::Point LeftLinePoint2(-678, 1207);
        cv::Point RightLinePoint1(1405, 945);
        cv::Point RightLinePoint2(2522, 1207);

        BoxPass_flag = CheckBoxInArea(RightLinePoint1, RightLinePoint2, LeftLinePoint1, LeftLinePoint2, box.x1, box.y2, box.x2, box.y2);
    }
    else if (camOrder == 2){
        BoxPass_flag = false; 
    }

    if (BoxPass_flag)
    {
        boxPoint = de.Get3dBBox(box.x1, box.y1, box.x2, box.y2, box.label, camId);
        camInfo.u = box.x1;
        camInfo.v = box.y1;
        camInfo.width = box.x2 - box.x1;
        camInfo.height = box.y2 - box.y1;
        camInfo.prob = box.prob;

        detObj.classId = translate_label(box.label);
        detObj.bPoint = boxPoint;
        detObj.camInfo = camInfo;
        detObj.fusionSourceId = 0;
    }
    return detObj;
}

void* run_yolo(void* ){
std::cout << __FILE__ << __LINE__ << std::endl;
std::cout << "run_inference start" << std::endl;
    std::vector<std_msgs::Header> headers_tmp;
    std::vector<std::vector<ITRI_Bbox>* > vbbx_output_tmp;
    std::vector<cv::Mat*> matSrcs_tmp;
    std::vector<uint32_t> matOrder_tmp;
    std::vector<uint32_t> matId_tmp;

    cv::Mat M_display;
    cv::Mat M_display_tmp;
    std::vector<cv::Scalar> cls_color = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0) , cv::Scalar(125, 125, 125)};
    cv::Scalar class_color;

    ros::Rate r(30);
    while(ros::ok() && !isInferStop)
    {
        pthread_mutex_lock(&mtxInfer);
        if(!isInferData) pthread_cond_wait(&cndInfer, &mtxInfer);
        pthread_mutex_unlock(&mtxInfer);
        // copy data
        headers_tmp = headers;
        vbbx_output_tmp = vbbx_output;
        matSrcs_tmp = matSrcs;
        matOrder_tmp = matOrder;
        matId_tmp = matId;

        // reset data
        headers.clear();
        matSrcs.clear();
        matOrder.clear();
        matId.clear();
        vbbx_output.clear();
        vBBX60_0.clear();
        vBBX60_1.clear();
        vBBX60_2.clear();
        isInferData = false;
        isInferData_0 = false;
        isInferData_1 = false;
        isInferData_2 = false;

        if (!input_resize) yoloApp.input_preprocess(matSrcs_tmp); 
        else yoloApp.input_preprocess(matSrcs_tmp, matId_tmp, input_resize); 

        yoloApp.inference_yolo(); 
        yoloApp.get_yolo_result(&matOrder_tmp, vbbx_output_tmp);

        /**
         *      publish results here
         */ 
        msgs::DetectedObjectArray doa;
        std::vector<msgs::DetectedObject> vDo;
        for(uint32_t ndx = 0; ndx < vbbx_output_tmp.size(); ndx++){
            std::vector<ITRI_Bbox>* tmpBBx = vbbx_output_tmp[ndx];
            if(imgResult_publish || display_flag)  
            {
                cv::resize((*matSrcs_tmp[ndx]), M_display_tmp, cv::Size(rawimg_w, rawimg_h) , 0, 0, 0);
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
                if(detObj.bPoint.p0.x != 0 && detObj.bPoint.p0.z != 0)
                    vDo.push_back(detObj); 
                if(display_flag)
                {   
                    int distMeter_p0x = detObj.bPoint.p0.x;
                    int distMeter_p0y = detObj.bPoint.p0.y;
                    int x1 = detObj.camInfo.u;
                    int x2 = detObj.camInfo.u + detObj.camInfo.width;
                    int y2 = detObj.camInfo.v + detObj.camInfo.height;
                    int distMeter_p3x = detObj.bPoint.p3.x;
                    int distMeter_p3y = detObj.bPoint.p3.y;
                    cv::putText(M_display, std::to_string(distMeter_p0x) + "," + std::to_string(distMeter_p0y), cvPoint(x1 - 100, y2 + 10), 0, 1, class_color, 2);
                    cv::putText(M_display, std::to_string(distMeter_p3x) + "," + std::to_string(distMeter_p3y), cvPoint(x2 + 10, y2 + 10), 0, 1, class_color, 2);
                }
            }

            doa.header = headers_tmp[ndx];
            doa.header.frame_id = "lidar"; //mapping to lidar coordinate
            doa.objects = vDo;

			if(cam_order == 0) {

                if (standard_FPS == 1) doa60_0 = doa;
                else pub60_0.publish(doa);

                if(imgResult_publish || display_flag) {
                    mat60_0_display = M_display.clone();

                    if(imgResult_publish){
                        image_publisher(mat60_0_display, headers_tmp[ndx], 0);
                    }
                }
			}else if(cam_order == 1) {	
                if (standard_FPS == 1) doa60_1 = doa;		
                else pub60_1.publish(doa);

                if(imgResult_publish || display_flag) {
                    mat60_1_display = M_display.clone();

                    if(imgResult_publish){
                        image_publisher(mat60_1_display, headers_tmp[ndx], 1);
                    }
                }
			}else if(cam_order == 2) {
                if (standard_FPS == 1) doa60_2 = doa;
                else pub60_2.publish(doa);

                if(imgResult_publish || display_flag) 
                {               
                    mat60_2_display = M_display.clone(); 

                    if(imgResult_publish)
                    {
                        image_publisher(mat60_2_display, headers_tmp[ndx], 2);
                    }    
                }           
			}
        }

        // reset data
        headers_tmp.clear();
        matSrcs_tmp.clear();
        matOrder_tmp.clear();
        matId_tmp.clear();
        vbbx_output_tmp.clear();
        r.sleep();
    }
std::cout << __FILE__ << __LINE__ << std::endl;
std::cout << "run_inference close" << std::endl;
    pthread_exit(0);
}

void* run_display(void* ){
    std::cout << "run_display start" << std::endl;
    cv::namedWindow("RightSide-60", cv::WINDOW_NORMAL);
    cv::namedWindow("Center-60", cv::WINDOW_NORMAL);
    cv::namedWindow("LeftSide-60", cv::WINDOW_NORMAL);

    int marker_h = 0;
    if (car_id == 1) marker_h = 590;
    else if (car_id == 1) marker_h = 584;    

    cv::Point BoundaryMarker1, BoundaryMarker2, BoundaryMarker3, BoundaryMarker4;
    BoundaryMarker1 = cv::Point(img_w/2 + 20, marker_h);
    BoundaryMarker2 = cv::Point(img_w/2 - 20, marker_h);
    BoundaryMarker3 = cv::Point(img_w/2, marker_h + 20);
    BoundaryMarker4 = cv::Point(img_w/2, marker_h - 20);

    ros::Rate r(10);
	while(ros::ok() && !isInferStop)
    {
        if (mat60_0_display.cols*mat60_0_display.rows == rawimg_size && mat60_1_display.cols*mat60_1_display.rows == rawimg_size && mat60_2_display.cols*mat60_2_display.rows == rawimg_size)
        { 
            cv::line(mat60_1_display, BoundaryMarker1, BoundaryMarker2, cv::Scalar(255, 255, 255), 1);
            cv::line(mat60_1_display, BoundaryMarker3, BoundaryMarker4, cv::Scalar(255, 255, 255), 1);
            cv::imshow("RightSide-60", mat60_0_display);
            cv::imshow("Center-60", mat60_1_display);
            cv::imshow("LeftSide-60", mat60_2_display);
            cv::waitKey(1);
        }
        r.sleep();
	}
        std::cout << "run_display close" << std::endl;
	pthread_exit(0);
}