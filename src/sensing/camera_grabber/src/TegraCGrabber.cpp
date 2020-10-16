#include <gst/gst.h> 
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/highgui.hpp>


#include "TegraCGrabber.h"
#include "grabber_args_parser.h"
#include "camera_params.h"






namespace SensingSubSystem
{
///
/// Class TegraCGrabber
///
TegraCGrabber::TegraCGrabber()
  : canvas(cam_ids_.size())
  , display_(&camera_buffer_)  
  , npp8u_ptrs_(cam_ids_.size())
  , resizer_(camera::raw_image_height, camera::raw_image_width, camera::image_height,
             camera::image_width)  // 720,1280 to 342,608
  , num_src_bytes_(camera::raw_image_height * camera::raw_image_width * 3)
  , ros_image(n)
  , video_capture_list(cam_ids_.size())   
{
  InitParameters();
}

void TegraCGrabber::InitParameters()
{

  //----- initiate driver --------- 
  std::string camera_grabber_pkg_path = ros::package::getPath("camera_grabber");
  //std::cout << "camera grabber dir : " << camera_grabber_pkg_path << std::endl;

  std::string ar0231_sh_directory_path = camera_grabber_pkg_path + "/src/CameraGrabber";
  //std::cout << "ar0231_sh_directory_path : " << ar0231_sh_directory_path << std::endl;

  std::string ar0231_sh_file_path = camera_grabber_pkg_path + "/src/CameraGrabber/init_ar0231.sh";
  //std::cout << "ar0231_sh_file_path : " << ar0231_sh_file_path << std::endl;

  // password selection
  auto password = SensingSubSystem::get_password();

  std::string ar0231_sh_call_command = "sh " + ar0231_sh_file_path + " " + ar0231_sh_directory_path + " " + password;
  std::cout << "ar0231_sh_call_command : " << ar0231_sh_call_command << std::endl;

  int n = ar0231_sh_call_command.length(); 
  
  // declaring character array 
  char cmd_array[n + 1]; 
  
  // copying the contents of the string to char array 
  strcpy(cmd_array, ar0231_sh_call_command.c_str()); 
 
  int ret = system(cmd_array);
  if ( ret == 0 )
    std::cout <<"ar0231 camera driver initiate OK" << std::endl;
  else
    std::cout <<"ar0231 camera driver initiate fail" << std::endl;

}

TegraCGrabber::~TegraCGrabber()
{
/* do nothing */
}




//********************
// Gstreamer funcstion
//********************

bool TegraCGrabber::gst_pipeline_init(int video_index)
{ 
  char caps[300];
   
  memset(caps, 0, sizeof(caps));
  if(resize_)
  {  //resize
    sprintf(caps,
            "v4l2src device=/dev/video%d ! "
            "videoconvert ! videoscale ! "
            "video/x-raw, width=%d, height=%d, format=BGR ! "
            "appsink",
            video_index, camera::image_width, camera::image_height
            );
    
  }
  else
  { // normal
    sprintf(caps,
            "v4l2src device=/dev/video%d ! " 
            "video/x-raw, width=%d, height=%d, format=UYVY ! "           
            "videoconvert ! "
            "video/x-raw, width=%d, height=%d, format=BGR ! "
            "appsink",
            video_index, 
            camera::raw_image_width, camera::raw_image_height, 
            camera::raw_image_width, camera::raw_image_height
            );

  }

  cv::VideoCapture capture(caps);
  if(!capture.isOpened()) 
  {
    std::cout<< "Failed to open camera " << video_index << " fail " << std::endl;
    return false;
  }
  else
  {
    unsigned int width  = capture.get(cv::CAP_PROP_FRAME_WIDTH); 
    unsigned int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT); 
    unsigned int fps    = capture.get(cv::CAP_PROP_FPS);
    unsigned int pixels = width*height;
    std::cout <<"<<cam" << video_index << " open success>>" << std::endl;
    std::cout <<"  Frame size : "<<width<<" x "<<height<<", "<<pixels<<" Pixels "<<fps<<" FPS"<<std::endl;

  }

  video_capture_list[video_index] = (capture);

  return true;
}

bool TegraCGrabber::initializeModulesGst(const bool do_resize)
{
  for (const auto cam_id : cam_ids_)
  {
    ros_image.add_a_pub(cam_id, camera::topics[cam_id]);
  }

  
  camera_buffer_.initBuffer();
  resize_ = do_resize;
  

  //Gstreamer    
  for(unsigned int index = 0; index < cam_ids_.size(); index++)
  {
    if(gst_pipeline_init(index) == false)
    {
          std::cout << "initializeModulesGst init camera "<< index << " fail!\n" << std::endl;
          return false;
    }
  }


  std::cout << "initializeModulesGst init done!\n" << std::endl;
  return true;
}


bool TegraCGrabber::runPerceptionGst()
{
  auto fps = SensingSubSystem::get_expected_fps();
  ros::Rate loop_rate(fps);  
  bool ret;
  bool for_running = true;  
  int i;
  int height, width;
  
  if(resize_)
  { 
     height = camera::image_height;
     width = camera::image_width;
  }
  else
  { //normal
     height = camera::raw_image_height;
     width = camera::raw_image_width;
  }

  //create green screen mat, (0, 131, 0) mean green color 
  cv::Mat green_mat(height, width, CV_8UC3, cv::Scalar(0, 131,0));  
  cv::Mat diff(height, width, CV_8UC3, cv::Scalar(0, 0,0));  
  cv::Mat diffcolor(height, width, CV_8UC1);

 


  cv::namedWindow("MyCameraPreview", cv::WindowFlags::WINDOW_AUTOSIZE);
  
  

  while (ros::ok())
  {
    // update vehicle state & imu state
    ros::spinOnce();
          
    int cam_count = cam_ids_.size();


   
    for (i = 0; i < cam_count ; ++i) 
    {
      if(for_running == false)
         break;  
      
      // grab frame from camera
      ret = video_capture_list[i].read(canvas[i]);
     
      // check the frame whether green screen, green screen mean camera read fail
      cv::compare(green_mat, canvas[i], diff, cv::CMP_NE);      

      // countNonZero only available when single channel, conver 3 channel to 1 channel     
      cv::cvtColor(diff, diffcolor, CV_BGR2GRAY, 1);      

      // The nz is 0 when frame is green screen
      int nz = cv::countNonZero(diffcolor);

      
      if ( (!ret) || (nz == 0) )  
      {
        if(for_running)
        {
          std::cout << "ERROR : video_capture_list read camera " << i << " fail \n" <<std::endl;
          std::cout << "Please press CTRL+C to break program \n" <<std::endl;
          for_running = false; // stop the for loop
        }
      }
      else
      {

        //cv::imshow("MyCameraPreview", canvas[i]);
        //cv::waitKey(1); // let imshow draw

        ros_image.send_image_rgb_gstreamer(cam_ids_[i], canvas[i]);
      }      
      
      

    }

    loop_rate.sleep();

  }  // end while

  return true;
}
}  // namespace SensingSubSystem
