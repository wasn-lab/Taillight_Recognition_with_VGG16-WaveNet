// ----------------------- OpenPose C++ API Tutorial - Example 3 - Body from image -----------------------
// It reads an image, process it, and displays it with the pose (and optionally hand and face) keypoints. In addition,
// it includes all the OpenPose configuration flags (enable/disable hand, face, output saving, etc.).

// Third-party dependencies
#include <opencv2/opencv.hpp>
// Command-line user interface
//#define OPENPOSE_FLAGS_DISABLE_PRODUCER
//#define OPENPOSE_FLAGS_DISABLE_DISPLAY
//#include <openpose/flags.hpp>
// OpenPose dependencies
//#include <openpose/headers.hpp>
#include <math.h>
#include <algorithm>
#include <string>
#include <vector>

#define M_PIl 3.141592653589793238462643383279502884L /* pi */
class Get_keypoints_and_predict
{
  /*public:
        double Get_distance2(double x1, double y1, double x2, double y2)
        {
            return sqrt(pow(x1-x2, 2) + pow(y1-y2, 2));
        }
        double Get_angle2(double x1, double y1, double x2, double y2)
        {
            return M_PI/2 - atan2(abs(y1-y2), abs(x1-x2));
        }
        double* Get_triangle_angle(double x1, double y1, double x2, double y2, double x3, double y3)
        {
            double a = Get_distance2(x1, y1, x2, y2);
            double b = Get_distance2(x2, y2, x3, y3);
            double c = Get_distance2(x1, y1, x3, y3);
            double test = (a*a + c*c - b*b)/(2*a*c);
            static double angle[3] = {0.0f, 0.0f, 0.0f};
            if (test  <= 1 && test >= -1)
            {
                angle[0] = acos((a*a + c*c - b*b)/(2*a*c));
                angle[1] = acos((a*a + b*b - c*c)/(2*a*b));
                angle[2] = M_PI - angle[0] - angle[1];
            }
            else
            {
                if (std::max(a, std::max(b, c)) == a)
                    angle[2] = M_PI;
                else if (std::max(a, std::max(b, c)) == b)
                    angle[0] = M_PI;
                else
                    angle[1] = M_PI;
            }
            return angle;
        }

    int check_point(double posekeypoints[][25][2]){
       int max_point_index = 0,cnt = 0,max_cnt = 14;index = sizeof(posekeypoints)/(sizeof(posekeypoints[0]);
       for(int i = 0; i<index; i++){
           for(int j=1; j<=14; j++){
               if(posekeypoints[i][j][0]==0.0f&&posekeypoints[i][j][1]==0.0f){
                    cnt++;
               }
           }
           if(max_cnt >= cnt){
               max_point_index = i;
               max_cnt = cnt;
           }
           cnt = 0;
       }
       return max_point_index;
    }

    double* printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr,
    std::vector<double> feature)
    {
        try
        {
            // Example: How to use the pose keypoints
            if (datumsPtr != nullptr && !datumsPtr->empty())
            {
                std::vector<double> keypoints_x;
                std::vector<double> keypoints_y;
                    const auto& poseKeypoints = datumsPtr->at(0)->poseKeypoints;
                    //return max_point of multi-people in bbox
                    int max_point_index = check_point(poseKeypoints);
                    //Get body we need
                    int body_part[13] = {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14};
                    for(int i=0; i<13; i++)
                    {
                        keypoints_x.insert(keypoints_x.end(),poseKeypoints[max_point_index, body_part[i], 0]);
                        keypoints_y.insert(keypoints_y.end(),poseKeypoints[max_point_index, body_part[i], 1]);

                    }

                    //Caculate the features
                // op::printTime(opTimer, "Start caculating features. Total time: ", " seconds.", op::Priority::High);
                    int keypoints_num = 13;
                    for(int m=0; m<keypoints_num; m++)
                    {
                        for(int n=m+1; n<keypoints_num; n++)
                        {
                            double dist_x, dist_y, dist, angle;
                            if(keypoints_x[m]!=0.0f &&  keypoints_y[m]!=0.0f && keypoints_x[n]!=0.0f &&
    keypoints_y[n]!=0.0f)
                            {
                                dist_x = abs(keypoints_x[m]-keypoints_x[n]);
                                dist_y = abs(keypoints_y[m]-keypoints_y[n]);
                                dist = Get_distance2(keypoints_x[m], keypoints_y[m], keypoints_x[n], keypoints_y[n]);
                                angle = Get_angle2(keypoints_x[m], keypoints_y[m], keypoints_x[n], keypoints_y[n]);
                            }
                            else
                            {
                                dist_x = 0.0f;
                                dist_y = 0.0f;
                                dist = 0.0f;
                                angle = 0.0f;
                            }
                            double input[] = {dist_x,dist_y,dist,angle};
                            feature.insert(feature.end(),input,input + sizeof(input) / sizeof(input[0]));

                        }
                    }

                // op::printTime(opTimer, "Finish caculating 50/% features. Total time: ", " seconds.",
    op::Priority::High);
                    for(int m=0; m<keypoints_num; m++)
                    {
                        for(int n=m+1; n<keypoints_num; n++)
                        {
                            for(int k=n+1; k<keypoints_num; k++)
                            {
                                double angle[3] = {0.0f, 0.0f, 0.0f};
                                double* angle_ptr;
                                if(keypoints_x[m]!=0.0f &&  keypoints_y[m]!=0.0f && keypoints_x[n]!=0.0f &&
    keypoints_y[n]!=0.0f)
                                {

                                    angle_ptr = Get_triangle_angle(keypoints_x[m], keypoints_y[m], keypoints_x[n],
    keypoints_y[n], keypoints_x[k], keypoints_y[k]);
                                    angle[0] = *angle_ptr;
                                    angle[1] = *(angle_ptr+1);
                                    angle[2] = *(angle_ptr+2);
                                }
                                feature.insert(feature.end(), angle, angle + sizeof(angle) / sizeof(angle[0]));
                            }
                        }
                    }

                // op::printTime(opTimer, "Finish caculating features. Total time: ", " seconds.", op::Priority::High);
                    static double feature_arr[1174];
                    std::copy(feature.begin(), feature.end(), feature_arr);
                    return feature_arr;
                }
            else
                {
                // op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
                    return 0;
                }
        }
        catch (const std::exception& e)
        {
            // op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }


    int get_keypoints(cv::Mat cvImageToProcess, double x1, double y1, double x2, double y2)
    {
        try
        {
            // op::opLog("Starting OpenPose demo...", op::Priority::High);

            // Configuring OpenPose
            // op::opLog("Configuring OpenPose...", op::Priority::High);
            // op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
            // configureWrapper(opWrapper);

            // Starting OpenPose
            // op::opLog("Starting thread(s)...", op::Priority::High);
            // opWrapper.start();

                // cut image
                cv::Rect bbox(x1,y1,x2-x1,y2-y1);//x,y,width,height
                cv::Mat image_cut = cv::Mat(cvImageToProcess, bbox);
                cv::Mat image_copy = image_cut.clone();
                //display test_image

                // Process and display image
            const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(image_copy);
            // op::printTime(opTimer, "Openpose start. Total time: ", " seconds.", op::Priority::High);
            auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
            // op::printTime(opTimer, "Openpose end. Total time: ", " seconds.", op::Priority::High);
            double* feature_ptr;
                if (datumProcessed != nullptr)
            {
                    //std::cout<< *datumProcessed << std::endl;
                std::vector<double> feature;
                    double bbox[] = {x1,y1,x2,y2};
                    feature.insert(feature.end(),bbox,bbox + sizeof(bbox) / sizeof(bbox[0]));
                    feature_ptr = printKeypoints(datumProcessed, feature);

                    double feature_arr[1174];
                    cv::Mat feature_mat =cv::Mat(1, 1174, CV_32F, feature_arr);
                    predict(feature_mat);
            }
            else{}
            return 0;
        }
        catch (const std::exception&)
        {
            return -1;
        }
    }

    void predict(cv::Mat input_data)
    {
        // op::printTime(opTimer, "Start load model. Total time: ", " seconds.", op::Priority::High);
      cv::Ptr<cv::ml::RTrees> rf =cv::ml::StatModel::load<cv::ml::RTrees>("/home/wasn/rf_1frames_0.yml");
        // op::printTime(opTimer, "Alredy load model. Total time: ", " seconds.", op::Priority::High);
      float p =rf->predict(input_data);
        // op::printTime(opTimer, "Get predict result. Total time: ", " seconds.", op::Priority::High);
      std::cout<<"prediction: "<<p<<std::endl;
    }
        */
};

/*

int main(int argc, char *argv[])
{

    // Parsing command line flags
    // gflags::ParseCommandLineFlags(&argc, &argv, true);

    Get_keypoints_and_predict get_keypoints_and_predict;
  // Running tutorialApiCpp
    //std::vector<double> a (4, 5.0f);
    //double arr[4];
    //std::copy(a.begin(), a.end(), arr);
    //cv::Mat mat =cv::Mat(1, 4, CV_32F, arr);
    //get_keypoints_and_predict.predict(mat);
    std::string image_path = "/mnt/sda/JAAD/images/video_0001/00000.png";
  const cv::Mat cvImageToProcess = cv::imread(image_path);
    get_keypoints_and_predict.get_keypoints(cvImageToProcess, 1398.0f, 654.0f, 1486.0f, 892.0f);

    return 0;
}*/