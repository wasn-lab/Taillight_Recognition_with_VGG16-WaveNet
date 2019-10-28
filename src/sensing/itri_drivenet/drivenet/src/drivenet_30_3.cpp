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

#include "npp_wrapper_dn.h"
#include "npp_rotater_dn.h"
#include "drivenet/trt_yolo_interface.h"
#include "drivenet/DistanceEstimation.h"
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

/// launch param
int truck_id = 1;
int standard_FPS = 0;
int display_flag = 0;
int input_resize = 0; //grabber input mode 0: 1920x1208, 1:608x608 yolo format
bool imgResult_publish = 0; //publish 2d result image or not.

pthread_mutex_t mtxInfer;
pthread_cond_t cndInfer;

ros::Subscriber cam30_0;
ros::Subscriber cam30_1;
ros::Subscriber cam30_2;
std::string cam30_0_topicName;
std::string cam30_1_topicName;
std::string cam30_2_topicName;
image_transport::Publisher pubImg_30_0;
image_transport::Publisher pubImg_30_1;
image_transport::Publisher pubImg_30_2;
sensor_msgs::ImagePtr imgMsg;

int rawimg_w = 1920;
int rawimg_h = 1208;
int img_w = 1920;
int img_h = 1208;
int img_size = img_w*img_h;
int rawimg_size = rawimg_w*rawimg_h;
cv::Mat mat30_0;
cv::Mat mat30_0_rot;
cv::Mat mat30_0_display;
std::vector<ITRI_Bbox> vBBX30_0;

cv::Mat mat30_1;
cv::Mat mat30_1_display;
std::vector<ITRI_Bbox> vBBX30_1;

cv::Mat mat30_2;
cv::Mat mat30_2_rot;
cv::Mat mat30_2_display;
std::vector<ITRI_Bbox> vBBX30_2;

std::vector<cv::Mat*> matSrcs;
std::vector<uint32_t> matOrder;
std::vector<uint32_t> matId;
std::vector<std_msgs::Header> headers;

std::vector<std::vector<ITRI_Bbox>* > vbbx_output;
DistanceEstimation de;
Yolo_app yoloApp;
DriveNet_npp::NPPRotater *rotater1;
DriveNet_npp::NPPRotater *rotater2;

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
        mat30_0_display = cv::Mat(img_w, img_h, CV_8UC3, cv::Scalar(0));
        mat30_1_display = cv::Mat(img_h, img_w, CV_8UC3, cv::Scalar(0));
        mat30_2_display = cv::Mat(img_w, img_h, CV_8UC3, cv::Scalar(0));
    }

}

void sync_inference(int camOrder, int camId, std_msgs::Header& header, cv::Mat *mat, std::vector<ITRI_Bbox>* vbbx){
    pthread_mutex_lock(&mtxInfer);

    bool isPushData = false;
    if (camOrder == 0 && !isInferData_0) {isInferData_0 = true; isPushData = true;}
    if (camOrder == 1 && !isInferData_1) {isInferData_1 = true; isPushData = true;}
    if (camOrder == 2 && !isInferData_2) {isInferData_2 = true; isPushData = true;}

    if (isPushData){
        matSrcs.push_back(mat);
        matOrder.push_back(camOrder);
        matId.push_back(camId);
        vbbx_output.push_back(vbbx);
        headers.push_back(header);
        std::cout << __FILE__ << __LINE__ << ", camOrder: " << camOrder << std::endl;
    }

    if(matOrder.size() == 3){
        isInferData = true;
        pthread_cond_signal(&cndInfer);
    }

    pthread_mutex_unlock(&mtxInfer);
    
    while(isInferData) usleep(5);
    
}

void callback_30_0(const sensor_msgs::Image::ConstPtr &msg){
    cv_bridge::CvImageConstPtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    mat30_0 = cv_ptr->image;

    cv::Mat rot_tmp;
#ifdef NPP
    rotater2->rotate(mat30_0, rot_tmp, 270);
#else  
    cv::rotate(mat30_0, rot_tmp, cv::ROTATE_90_CLOCKWISE);
#endif
    mat30_0_rot = rot_tmp;

    std_msgs::Header h = msg->header;
    if(!isInferData_0) sync_inference(0, 7, h, &mat30_0_rot, &vBBX30_0);
}

void callback_30_1(const sensor_msgs::Image::ConstPtr &msg){
    cv_bridge::CvImageConstPtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    mat30_1 = cv_ptr->image;

    std_msgs::Header h = msg->header;
    if(!isInferData_1)  sync_inference(1, 8, h, &mat30_1, &vBBX30_1);
}

void callback_30_2(const sensor_msgs::Image::ConstPtr &msg){
    cv_bridge::CvImageConstPtr cv_ptr =
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    mat30_2 = cv_ptr->image;

    cv::Mat rot_tmp;
#ifdef NPP
    rotater1->rotate(mat30_2, rot_tmp, 90);
#else  
    cv::rotate(mat30_2, rot_tmp, cv::ROTATE_90_COUNTERCLOCKWISE);
#endif
    mat30_2_rot = rot_tmp;

    std_msgs::Header h = msg->header;
    if(!isInferData_2) sync_inference(2, 9, h, &mat30_2_rot, &vBBX30_2);
}

void callback_30_0_decode(sensor_msgs::CompressedImage compressImg){
    cv::imdecode(cv::Mat(compressImg.data),1).copyTo(mat30_0);
    cv::Mat rot_tmp;
#ifdef NPP
    rotater2->rotate(mat30_0, rot_tmp, 270);
#else  
    cv::rotate(mat30_0, rot_tmp, cv::ROTATE_90_CLOCKWISE);
#endif
    mat30_0_rot = rot_tmp;
    if(!isInferData_0) sync_inference(0, 7, compressImg.header, &mat30_0_rot, &vBBX30_0);
}

void callback_30_1_decode(sensor_msgs::CompressedImage compressImg){
    cv::imdecode(cv::Mat(compressImg.data),1).copyTo(mat30_1);
    
    if(!isInferData_1) sync_inference(1, 8, compressImg.header, &mat30_1, &vBBX30_1);
}

void callback_30_2_decode(sensor_msgs::CompressedImage compressImg){
    cv::imdecode(cv::Mat(compressImg.data),1).copyTo(mat30_2);
    cv::Mat rot_tmp;
#ifdef NPP
    rotater1->rotate(mat30_2, rot_tmp, 90);
#else  
    cv::rotate(mat30_2, rot_tmp, cv::ROTATE_90_COUNTERCLOCKWISE);
#endif
    mat30_2_rot = rot_tmp;
    if(!isInferData_2) sync_inference(2, 9, compressImg.header, &mat30_2_rot, &vBBX30_2);

}

void image_publisher(cv::Mat image, std_msgs::Header header, int camOrder)
{
    imgMsg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();

    if(camOrder == 0)
	    pubImg_30_0.publish(imgMsg);
    else if (camOrder == 1)
 	    pubImg_30_1.publish(imgMsg);
    else if (camOrder == 2)
	    pubImg_30_2.publish(imgMsg);
}
ros::Publisher pub30;
/// roslaunch drivenet drivenet30.launch
int main(int argc, char **argv)
{
    ros::init(argc, argv, "drivenet_30_3");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);

    isInferStop = false;
    isInferData = false;
    
    if (ros::param::get(ros::this_node::getName()+"/truck_id", truck_id));
	if (ros::param::get(ros::this_node::getName()+"/standard_fps", standard_FPS));
	if (ros::param::get(ros::this_node::getName()+"/display", display_flag));
	if (ros::param::get(ros::this_node::getName()+"/input_resize", input_resize));
	if (ros::param::get(ros::this_node::getName()+"/imgResult_publish", imgResult_publish));

    if (!input_resize)
    {
        cam30_0_topicName = "gmsl_camera/port_c/cam_0";
        cam30_1_topicName = "gmsl_camera/port_c/cam_1";
        cam30_2_topicName = "gmsl_camera/port_c/cam_2";
        
        cam30_0 = nh.subscribe(cam30_0_topicName + std::string("/image_raw/compressed"), 1, callback_30_0_decode);
        cam30_1 = nh.subscribe(cam30_1_topicName + std::string("/image_raw/compressed"), 1, callback_30_1_decode);
        cam30_2 = nh.subscribe(cam30_2_topicName + std::string("/image_raw/compressed"), 1, callback_30_2_decode);

        if(imgResult_publish)
        {
            pubImg_30_0 = it.advertise(cam30_0_topicName + std::string("/detect_image"), 1);
            pubImg_30_1 = it.advertise(cam30_1_topicName + std::string("/detect_image"), 1);
            pubImg_30_2 = it.advertise(cam30_2_topicName + std::string("/detect_image"), 1);
        }
    }
    else
    {
        cam30_0_topicName = "gmsl_camera/8";
        cam30_1_topicName = "gmsl_camera/9";
        cam30_2_topicName = "gmsl_camera/10";
        cam30_0 = nh.subscribe(cam30_0_topicName, 1, callback_30_0);
        cam30_1 = nh.subscribe(cam30_1_topicName, 1, callback_30_1);
        cam30_2 = nh.subscribe(cam30_2_topicName, 1, callback_30_2);
        
        if(imgResult_publish)
        {
            pubImg_30_0 = it.advertise(cam30_0_topicName + std::string("/detect_image"), 1);
            pubImg_30_1 = it.advertise(cam30_1_topicName + std::string("/detect_image"), 1);
            pubImg_30_2 = it.advertise(cam30_2_topicName + std::string("/detect_image"), 1);
        }
    }

    pub30 = nh.advertise<msgs::DetectedObjectArray>("/DetectedObjectArray/cam30", 4);

    pthread_mutex_init(&mtxInfer,NULL);
    pthread_cond_init(&cndInfer,NULL);

    int ret = pthread_create(&thrdYolo, NULL, &run_yolo, NULL);    
    if (standard_FPS == 1)
        ret = pthread_create(&thrdInterp, NULL, &run_interp, NULL);
    if (display_flag == 1)
        int retDisplay = pthread_create(&thrdDisplay, NULL, &run_display, NULL); 

    std::string pkg_path = ros::package::getPath("drivenet");
    std::string cfg_file = "/yolo_30.cfg";
    image_init();    
    yoloApp.init_yolo(pkg_path, cfg_file);
    de.init(truck_id);
    rotater1 = new DriveNet_npp::NPPRotater(img_h, img_w, 90);
    rotater2 = new DriveNet_npp::NPPRotater(img_h, img_w, 270);

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
msgs::DetectedObjectArray doa30;
void* run_interp(void* ){
    std::cout << "run_interp start" << std::endl;
    ros::Rate r(30);
	while(ros::ok() && !isInferStop)
    {
		pub30.publish(doa30);
        r.sleep();
	}
        std::cout << "run_interp close" << std::endl;
	pthread_exit(0);
}

bool CheckBoxInArea(cv::Point RightLinePoint1, cv::Point RightLinePoint2, cv::Point LeftLinePoint1, cv::Point LeftLinePoint2, int object_x1, int object_y1, int object_x2, int object_y2)
{
    bool point1 = false;
    bool point2 = false;
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

    // printf("C1:%d, C3:%d, C5:%d, C7:%d\n", C1, C3, C5, C7);
    // printf("C2:%d, C4:%d, C6:%d, C8:%d\n", C2, C4, C6, C8);

    if (C1 < 0 && C3 > 0 && C5 > 0 && C7 < 0)
        point2 = true;
    else
        point2 = false;
    if (C2 < 0 && C4 > 0 && C6 > 0 && C8 < 0)
        point1 = true;  
    else
        point1 = false;      
    return point1 && point2;
    
}
msgs::DetectedObject run_dist(ITRI_Bbox box, int cam_order, int cam_id){
    msgs::DetectedObject detObj;
    msgs::BoxPoint boxPoint;
    msgs::CamInfo camInfo;
    double *diatance_pixel_array;
    bool BoxPass_flag = false;

    if (cam_order == 0)
    {
        ///boundary
        cv::Point LeftLinePoint1, LeftLinePoint2, RightLinePoint1, RightLinePoint2;
        if (truck_id == 1)
        {
            LeftLinePoint1 = cv::Point(85, 800);
            LeftLinePoint2 = cv::Point(121, 1920);
            RightLinePoint1 = cv::Point(460, 800);
            RightLinePoint2 = cv::Point(2264, 1920);
            BoxPass_flag = CheckBoxInArea(RightLinePoint1, RightLinePoint2, LeftLinePoint1, LeftLinePoint2, box.x1, box.y2, box.x2, box.y2);
        }
        else if (truck_id == 2)
        {
            LeftLinePoint1 = cv::Point(65, 514);//
            LeftLinePoint2 = cv::Point(318, 1920);//
            RightLinePoint1 = cv::Point(646, 514);//
            RightLinePoint2 = cv::Point(1837, 1920);//
            BoxPass_flag = CheckBoxInArea(RightLinePoint1, RightLinePoint2, LeftLinePoint1, LeftLinePoint2, box.x1, box.y2, box.x2, box.y2);
        }

        if (truck_id == 1) if (box.y2 < 800) BoxPass_flag = false;        
        if (truck_id == 2) if (box.y2 < 514) BoxPass_flag = false;  //      
    }
    else if (cam_order == 1)
    {
        ///boundary
        cv::Point LeftLinePoint1, LeftLinePoint2, RightLinePoint1, RightLinePoint2;
        if (truck_id == 1)
        {
            LeftLinePoint1 = cv::Point(675, 584);
            LeftLinePoint2 = cv::Point(-80, 1205);
            RightLinePoint1 = cv::Point(1346, 584);
            RightLinePoint2 = cv::Point(2211, 1205);
        }
        else if (truck_id == 2)
        {
            // LeftLinePoint1 = cv::Point(605, 555);
            // LeftLinePoint2 = cv::Point(-93, 1207);
            // RightLinePoint1 = cv::Point(1055, 555);
            // RightLinePoint2 = cv::Point(1800, 1207);

            // LeftLinePoint1 = cv::Point(814, 612);
            // LeftLinePoint2 = cv::Point(104, 1205);
            // RightLinePoint1 = cv::Point(1262, 612);
            // RightLinePoint2 = cv::Point(1854, 1205);

            LeftLinePoint1 = cv::Point(557, 555);
            LeftLinePoint2 = cv::Point(-404, 1207);
            RightLinePoint1 = cv::Point(1224, 555);
            RightLinePoint2 = cv::Point(2464, 1207);
        }
        BoxPass_flag = CheckBoxInArea(RightLinePoint1, RightLinePoint2, LeftLinePoint1, LeftLinePoint2, box.x1, box.y2, box.x2, box.y2);
        if (truck_id == 1) if (box.y2 > 722) BoxPass_flag = false;
        if (truck_id == 2) if (box.y2 > 700) BoxPass_flag = false;
        // if (truck_id == 2) if (box.y2 > 726) BoxPass_flag = false;
    }
    else if (cam_order == 2)
    {
        ///boundary
        cv::Point LeftLinePoint1, LeftLinePoint2, RightLinePoint1, RightLinePoint2;
        if (truck_id == 1)
        {
            LeftLinePoint1 = cv::Point(1029, 584);
            LeftLinePoint2 = cv::Point(-892, 1920);
            RightLinePoint1 = cv::Point(1395, 584);
            RightLinePoint2 = cv::Point(598, 1920);
            BoxPass_flag = CheckBoxInArea(RightLinePoint1, RightLinePoint2, LeftLinePoint1, LeftLinePoint2, box.x1, box.y2, box.x2, box.y2);
        }
        else if (truck_id == 2)
        {
            LeftLinePoint1 = cv::Point(891, 250);
            LeftLinePoint2 = cv::Point(-2067, 1803);
            RightLinePoint1 = cv::Point(1229, 250);
            RightLinePoint2 = cv::Point(873, 1803);
            BoxPass_flag = CheckBoxInArea(RightLinePoint1, RightLinePoint2, LeftLinePoint1, LeftLinePoint2, box.x1, box.y2, box.x2, box.y2);
        }
        if (truck_id == 1) if (box.y2 < 726) BoxPass_flag = false;
        if (truck_id == 2) if (box.y2 < 250) BoxPass_flag = false;
    }
    if (BoxPass_flag)
    {
        boxPoint = de.Get3dBBox(box.x1, box.y1, box.x2, box.y2, box.label, cam_id);

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
std::cout << "run_inference start" << std::endl;
std::cout << __FILE__ << __LINE__ << std::endl;
    std::vector<std_msgs::Header> headers_tmp;
    std::vector<ITRI_Bbox> vBBX60_0_tmp;
    std::vector<ITRI_Bbox> vBBX60_1_tmp;
    std::vector<ITRI_Bbox> vBBX60_2_tmp;
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
        vBBX30_0.clear();
        vBBX30_1.clear();
        vBBX30_2.clear();
        vbbx_output.clear();
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
                if (matOrder_tmp[ndx] == 0 || matOrder_tmp[ndx] == 2 )
                    cv::resize((*matSrcs_tmp[ndx]), M_display_tmp, cv::Size(rawimg_h, rawimg_w), 0, 0, 0);
                else
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
                if(detObj.bPoint.p0.x != 0 && detObj.bPoint.p0.z != 0) vDo.push_back(detObj);
                if(display_flag)
                {   
                    int distMeter_p0x, distMeter_p3x, distMeter_p0y, distMeter_p3y;
                    if (cam_order == 0 || cam_order == 2)
                    {
                        distMeter_p0x = detObj.bPoint.p7.x;
                        distMeter_p3x = detObj.bPoint.p4.y;
                        distMeter_p0y = detObj.bPoint.p7.y;  
                        distMeter_p3y = detObj.bPoint.p4.y;
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
            if(cam_order == 0) {
                if(imgResult_publish || display_flag) {
                    mat30_0_display = M_display.clone();
                    
                    if(imgResult_publish){
                        image_publisher(mat30_0_display, headers_tmp[ndx], 0);
                    }
                }                
            }else if(cam_order == 1) {
                if(imgResult_publish || display_flag) {
                    mat30_1_display = M_display.clone();  

                    if(imgResult_publish)
                    {
                        image_publisher(mat30_1_display, headers_tmp[ndx], 1);
                    }
                } 
            }else if(cam_order == 2) {
                if(imgResult_publish || display_flag) {
                    mat30_2_display = M_display.clone();   

                    if(imgResult_publish){
                        image_publisher(mat30_2_display, headers_tmp[ndx], 2);
                    }
                }                   
            }

        }
        doa.header = headers_tmp[2];
        doa.header.frame_id = "lidar";
        doa.objects = vDo;
        if (standard_FPS == 1) doa30 = doa;
        else pub30.publish(doa);

        // reset data
        headers_tmp.clear();
        matSrcs_tmp.clear();
        matId_tmp.clear();
        matOrder_tmp.clear();
        vbbx_output_tmp.clear();
        r.sleep();
    }
std::cout << __FILE__ << __LINE__ << std::endl;
std::cout << "run_inference close" << std::endl;
    pthread_exit(0);
}

void* run_display(void* ){
    std::cout << "run_display start" << std::endl;
    cv::namedWindow("LeftSide-30", CV_WINDOW_NORMAL);
    cv::namedWindow("Center-30", CV_WINDOW_NORMAL);
    cv::namedWindow("RightSide-30", CV_WINDOW_NORMAL);

    int marker_h_0 = 0, marker_h_1 = 0, marker_h_2 = 0;
    if (truck_id == 1) { marker_h_0 = 800; marker_h_1 = 584; marker_h_2 = 726; }
    else if (truck_id == 1) marker_h_1 = 612;  

    cv::Point BoundaryMarker1_0, BoundaryMarker2_0, BoundaryMarker3_0, BoundaryMarker4_0;
    cv::Point BoundaryMarker1_1, BoundaryMarker2_1, BoundaryMarker3_1, BoundaryMarker4_1;
    cv::Point BoundaryMarker1_2, BoundaryMarker2_2, BoundaryMarker3_2, BoundaryMarker4_2;
    BoundaryMarker1_0 = cv::Point(img_h/2 + 20, marker_h_0);
    BoundaryMarker2_0 = cv::Point(img_h/2 - 20, marker_h_0);
    BoundaryMarker3_0 = cv::Point(img_h/2, marker_h_0 + 20);
    BoundaryMarker4_0 = cv::Point(img_h/2, marker_h_0 - 20);
    BoundaryMarker1_1 = cv::Point(img_w/2 + 20, marker_h_1);
    BoundaryMarker2_1 = cv::Point(img_w/2 - 20, marker_h_1);
    BoundaryMarker3_1 = cv::Point(img_w/2, marker_h_1 + 20);
    BoundaryMarker4_1 = cv::Point(img_w/2, marker_h_1 - 20);
    BoundaryMarker1_2 = cv::Point(img_h/2 + 20, marker_h_2);
    BoundaryMarker2_2 = cv::Point(img_h/2 - 20, marker_h_2);
    BoundaryMarker3_2 = cv::Point(img_h/2, marker_h_2 + 20);
    BoundaryMarker4_2 = cv::Point(img_h/2, marker_h_2 - 20);

    ros::Rate r(10);
	while(ros::ok() && !isInferStop)
    {
        if (mat30_0_display.cols*mat30_0_display.rows == rawimg_size && mat30_1_display.cols*mat30_1_display.rows == rawimg_size && mat30_2_display.cols*mat30_2_display.rows == rawimg_size)
        {
            cv::line(mat30_0_display, BoundaryMarker1_0, BoundaryMarker2_0, cv::Scalar(255, 255, 255), 1);
            cv::line(mat30_0_display, BoundaryMarker3_0, BoundaryMarker4_0, cv::Scalar(255, 255, 255), 1);
            cv::line(mat30_1_display, BoundaryMarker1_1, BoundaryMarker2_1, cv::Scalar(255, 255, 255), 1);
            cv::line(mat30_1_display, BoundaryMarker3_1, BoundaryMarker4_1, cv::Scalar(255, 255, 255), 1);
            cv::line(mat30_2_display, BoundaryMarker1_2, BoundaryMarker2_2, cv::Scalar(255, 255, 255), 1);
            cv::line(mat30_2_display, BoundaryMarker3_2, BoundaryMarker4_2, cv::Scalar(255, 255, 255), 1);
            cv::imshow("LeftSide-30", mat30_0_display);
            cv::imshow("Center-30", mat30_1_display);
            cv::imshow("RightSide-30", mat30_2_display);
            cv::waitKey(1);
        }
        r.sleep();
	}
        std::cout << "run_display close" << std::endl;
	pthread_exit(0);
}
