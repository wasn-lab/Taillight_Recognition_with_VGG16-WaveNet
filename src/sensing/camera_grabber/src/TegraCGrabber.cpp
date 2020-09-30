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
             camera::image_width)  // 720,1280 to 384,608
  , num_src_bytes_(camera::raw_image_height * camera::raw_image_width * 3)
  , ros_image(n)
  , video_capture_list(cam_ids_.size())  
{
  InitParameters();
}

void TegraCGrabber::InitParameters()
{
  for (size_t i = 0; i < cam_ids_.size(); i++)
  {
    int dummy;
    npp8u_ptrs_[i] = nppiMalloc_8u_C3(camera::raw_image_width, camera::raw_image_height, &dummy);
  }
}

TegraCGrabber::~TegraCGrabber()
{
  for (size_t i = 0; i < cam_ids_.size(); i++)
  {
    nppiFree(npp8u_ptrs_[i]);
  }
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
            "video/x-raw, width=%d, height=%d, format=UYVY, framerate=22/1 ! "           
            "videoconvert ! "
            "video/x-raw, width=%d, height=%d, format=BGR ! "
            "appsink",
            video_index, 
            camera::raw_image_width_gstreamer, camera::raw_image_height_gstreamer, 
            camera::raw_image_width_gstreamer, camera::raw_image_height_gstreamer
            );
  }

  cv::VideoCapture capture(caps);
  if(!capture.isOpened()) 
  {
    std::cout<< "Failed to open camera." << std::endl;
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

void TegraCGrabber::initializeModulesGst(const bool do_resize, const bool do_crop)
{
  for (const auto cam_id : cam_ids_)
  {
    ros_image.add_a_pub(cam_id, camera::topics[cam_id]);
  }

  
  camera_buffer_.initBuffer();
  resize_ = do_resize;
  crop_ = do_crop;

  //Gstreamer    
  for(unsigned int index = 0; index < cam_ids_.size(); index++)
  {
    if(gst_pipeline_init(index) == false)
    {
          std::cout << "initializeModulesGst init fail!\n" << std::endl;
          return;
    }
  }


  std::cout << "initializeModulesGst init done!\n" << std::endl;
}


bool TegraCGrabber::runPerceptionGst()
{
  auto fps = SensingSubSystem::get_expected_fps();
  ros::Rate loop_rate(fps);

  cv::namedWindow("MyCameraPreview", cv::WindowFlags::WINDOW_AUTOSIZE);

  while (ros::ok())
  {
    // update vehicle state & imu state
    ros::spinOnce();
          
    int cam_count = cam_ids_.size();
/*
    if(!crop_)
    {
      cam_count=cam_ids_.size()-1;
    }
*/

#pragma omp parallel for    
    //for (size_t i = 0; i < cam_count; ++i)
    for (int i = 0; i < cam_count; ++i) //Jason
    {
      if(!video_capture_list[i].read(canvas[i]))
      {
        std::cout << "video_capture_list read video " << i << " fail \n" <<std::endl;
      }
      else
      {
/*
        cv::imshow("MyCameraPreview", canvas[i]);
        cv::waitKey(1); // let imshow draw
*/
        ros_image.send_image_rgb(cam_ids_[i], canvas[i]);
      }
    }

    loop_rate.sleep();

  }  // end while

  return true;
}
}  // namespace SensingSubSystem
