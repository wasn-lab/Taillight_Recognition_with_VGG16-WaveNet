#include <gst/gst.h>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/highgui.hpp>


#include "JetsonXavierGrabber.h"
#include "grabber_args_parser.h"
#include "camera_params.h"

//if MV_IMAGE_DEBUG defined, the motion_vector have to true in jetson_xavier_b1.launch
//#define MV_IMAGE_DEBUG 1


namespace SensingSubSystem
{
///
/// Class JetsonXavierGrabber
///
JetsonXavierGrabber::JetsonXavierGrabber()
  : canvas(cam_ids_.size())
  , canvas_tmp(cam_ids_.size())
  , ctx(cam_ids_.size())
  , cvBGR(cam_ids_.size())
  , cvYUV(cam_ids_.size())
  , cvBGR_queue(cam_ids_.size())
  , fuRes_enc(cam_ids_.size())
  , debug_counter(cam_ids_.size())
  , cvMV(cam_ids_.size())
  , mv_msgs_array(cam_ids_.size())
  , display_(&camera_buffer_)
  , npp8u_ptrs_(cam_ids_.size())
  , resizer_(camera::raw_image_height, camera::raw_image_width, camera::image_height,
             camera::image_width)  // 720,1280 to 342,608
  , remapper_(camera::raw_image_height, camera::raw_image_width)
  , ros_image(n)
  , video_capture_list(cam_ids_.size())
{
  char buf[16];

  // check camera driver whether loaded
  FILE *fd = popen("lsmod | grep ar0147", "r");

  if (fread (buf, 1, sizeof (buf), fd) > 0) // if there is some result the camera driver module must be loaded
  {
    printf ("camera driver module always is loaded\n");
  }
  else
  {
    printf ("camera driver module is not loaded\n");
    InitParameters();
  }

  pclose(fd);

}

void JetsonXavierGrabber::InitParameters()
{
  //----- initiate driver ---------
  std::string camera_grabber_pkg_path = ros::package::getPath("camera_grabber");
  // std::cout << "camera grabber dir : " << camera_grabber_pkg_path << std::endl;

  std::string ar0231_sh_directory_path = camera_grabber_pkg_path + "/src/CameraGrabber";
  // std::cout << "ar0231_sh_directory_path : " << ar0231_sh_directory_path << std::endl;

  std::string ar0231_sh_file_path = camera_grabber_pkg_path + "/src/CameraGrabber/init_ar0231_driver.sh";
  // std::cout << "ar0231_sh_file_path : " << ar0231_sh_file_path << std::endl;

  // password selection
  auto password = SensingSubSystem::get_password();

  // car_driver or laboratory camera driver
  const bool car_driver = SensingSubSystem::car_driver();

  std::string ar0231_sh_call_command; 
  if (car_driver)
    ar0231_sh_call_command = "sh " + ar0231_sh_file_path + " " + ar0231_sh_directory_path + " " + password + " " + "true";
  else
    ar0231_sh_call_command = "sh " + ar0231_sh_file_path + " " + ar0231_sh_directory_path + " " + password + " " + "false";
  std::cout << "ar0231_sh_call_command : " << ar0231_sh_call_command << std::endl;

  int n = ar0231_sh_call_command.length();

  // declaring character array
  char cmd_array[n + 1];

  // copying the contents of the string to char array
  strcpy(cmd_array, ar0231_sh_call_command.c_str());

  int ret = system(cmd_array);
  if (ret == 0)
    std::cout << "ar0231 camera driver initiate OK" << std::endl;
  else
    std::cout << "ar0231 camera driver initiate fail" << std::endl;
}

JetsonXavierGrabber::~JetsonXavierGrabber()
{
  for(unsigned int i=0; i<cam_ids_.size(); i++)
  {  
    video_capture_list[i].release();  //close video  
  }
  
  if (motion_vector_) 
  {
    for(unsigned int i=0; i<cam_ids_.size(); i++)
    {
      MvExtractor_Deinit(&ctx[i]);
    }
  }
}

//********************
// Gstreamer funcstion
//********************

bool JetsonXavierGrabber::gst_pipeline_init(int video_index)
{
  char caps[300];

  memset(caps, 0, sizeof(caps));

  // normal
  sprintf(caps, "v4l2src device=/dev/video%d ! "
                "video/x-raw, width=%d, height=%d, format=UYVY ! "
                "videoconvert ! "
                "video/x-raw, width=%d, height=%d, format=BGR ! "
                "appsink",
            video_index, camera::raw_image_width, camera::raw_image_height, camera::raw_image_width,
            camera::raw_image_height);
  

  cv::VideoCapture capture(caps, cv::CAP_GSTREAMER);

  
  if (!capture.isOpened())
  {
    sleep(1);
    if(capture.open(caps)) //try re-open again
    {
      std::cout << "(1) reopen camera success " << video_index << std::endl;
    }
    else
    {
      sleep(1);
      if(capture.open(caps)) //try re-open again
      {
        std::cout << "(2) reopen camera success " << video_index << std::endl;
      }
      else
      {
        std::cout << "Failed to reopen camera " << video_index << " fail " << std::endl;
        return false;
      }
    }
  }
  
  // success open camera
  unsigned int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  unsigned int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  unsigned int fps = capture.get(cv::CAP_PROP_FPS);
  unsigned int pixels;
  std::cout << "<<cam" << video_index << " open success>>" << std::endl;
  if (resize_)
  {
    pixels = camera::image_width * camera::image_height;
    std::cout << "  Frame size : " << camera::image_width << " x " << camera::image_height << ", " << pixels << " Pixels " << fps << " FPS"
            << std::endl;  
  }  
  else
  {
    pixels = width * height;
    std::cout << "  Frame size : " << width << " x " << height << ", " << pixels << " Pixels " << fps << " FPS"
            << std::endl;
  }
  

  video_capture_list[video_index] = (capture);

  return true;
}

bool JetsonXavierGrabber::initializeModulesGst(const bool do_resize)
{
  // get motion_vector_ variable from roslaunch file
  motion_vector_ = SensingSubSystem::motion_vector();

  resize_ = do_resize;
  
#ifdef MV_IMAGE_DEBUG
  if(!motion_vector_)
  {
    std::cout << "Error : MV_IMAGE_DEBUG defined only when motion_vector is true in launch file" << std::endl;
    return false;
  }
#endif

  for (const auto cam_id : cam_ids_)
  {
    ros_image.add_a_pub(cam_id, camera::topics[cam_id]);
    
    if (motion_vector_) 
    { 
#ifdef MV_IMAGE_DEBUG      
      ros_image.add_a_pub_mv(cam_id , camera::topics[cam_id]); 
#endif
      ros_image.add_a_pub_mv_msgs(cam_id , camera::topics[cam_id]);     
    }
  }

  camera_buffer_.initBuffer();
  

  sleep(1);
  // Gstreamer
  for (unsigned int index = 0; index < cam_ids_.size(); index++)
  {
    sleep(1);
    if (gst_pipeline_init(index) == false)
    {
      std::cout << "initializeModulesGst init camera " << index << " fail!\n" << std::endl;
      return false;
    }
  }  

  //MvExtractor init  
  if (motion_vector_) 
  {

    auto fps = SensingSubSystem::get_expected_fps();
    int ret = 0;
    for (unsigned int i = 0; i < cam_ids_.size(); i++)
    {      
      MvExtractor_Settings(&ctx[i]);
      
      ctx[i].width = camera::raw_image_width;
      ctx[i].height = camera::raw_image_height;
      ctx[i].fps_n = (int)fps;
      ctx[i].iframe_interval = 120;
      ctx[i].idr_interval = 120;
      ctx[i].stats = true;
      
      ret = MvExtractor_Init(&ctx[i], i);
      

      if (ret)
      {
        std::cout << "MvExtractor_Init failed for camera " << i << std::endl;
      }
      else
      {
        std::cout << "MvExtractor_Init success for camera " << i << std::endl;
      }

      debug_counter[i] = 0;

    }
  }


  std::cout << "initializeModulesGst init done!\n" << std::endl;
  return true;
}

bool JetsonXavierGrabber::runPerceptionGst()
{
  auto fps = SensingSubSystem::get_expected_fps();
  ros::Rate loop_rate(fps);
  bool ret[cam_ids_.size()];
  bool for_running = true;
  int i;
  int height, width;
  cv::Mat canvas_resize;
  cv::Mat canvas_undestortion;
  bool check_green_screen = true;
  msgs::MotionVectorArray mv_msgs_arry;
  
  
  // normal
  height = camera::raw_image_height;
  width = camera::raw_image_width;
  

  // create green screen mat, (0, 131, 0) mean green color
  cv::Mat green_mat(height, width, CV_8UC3, cv::Scalar(0, 131, 0));
  cv::Mat diff(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat diffcolor(height, width, CV_8UC1);

  

  while (ros::ok() && for_running)
  {
    // update vehicle state & imu state
    ros::spinOnce();

    int cam_count = cam_ids_.size();

    for (i = 0; i < cam_count; ++i)
    {
      if (for_running == false)
        break;

      if (motion_vector_) 
      {

        /* Check MvExtractor error flag */
        for(int i=0; i<cam_count; i++)
        {
            if(ctx[i].got_error || ctx[i].enc->isInError())
            {
                std::cerr << "ctx[i].got_error || ctx[i].enc->isInError()" << std::endl;
                break;
            }
        }
      }


      // grab frame from camera
      ret[i] = video_capture_list[i].read(canvas[i]); 
    }

    ros_time_ = ros::Time::now();

    for (i = 0; i < cam_count; i++)
    {
      if (for_running == false)
        break;

      if (check_green_screen)
      {    
        // check the frame whether green screen, green screen mean camera read fail
        cv::compare(green_mat, canvas[i], diff, cv::CMP_NE);

        // countNonZero only available when single channel, conver 3 channel to 1 channel
        cv::cvtColor(diff, diffcolor, CV_BGR2GRAY, 1);

        // The nz is 0 when frame is green screen
        int nz = cv::countNonZero(diffcolor);

        if ((!ret[i]) || (nz == 0))
        {
          if (for_running)
          {
            std::cout << "ERROR : video_capture_list read camera " << i << " fail \n" << std::endl;
            std::cout << "ERROR : camera " << i << " green screen \n" << std::endl;
            std::cout << "Please press CTRL+C to break program \n" << std::endl;
            for_running = false;  // stop for loop
          }
        }
      }  
      
      if (for_running)
      {     
        if (camera::distortion[i])
        { //FOV 120                  
          remapper_.remap(canvas[i], canvas_undestortion);//undistrotion , 1280x720

          if (motion_vector_)
          {
            cvBGR[i] = canvas_undestortion;
          }

          if (resize_)
          {
            resizer_.resize(canvas_undestortion, canvas_resize); //608x342
            canvas[i] = canvas_resize;            
          }
          else
          {
            canvas[i] = canvas_undestortion.clone();                        
          }
        }
        else
        {
          if (motion_vector_)
          {
            cvBGR[i] = canvas[i];
          }

          if (resize_)
          {
            resizer_.resize(canvas[i], canvas_resize); //608x342
            canvas[i] = canvas_resize;            
          }
        }

        canvas_tmp[i] = canvas[i].clone();

        if (motion_vector_) 
        {          
          //cvBGR[i] = canvas[i].clone();
          if(cvBGR[i].empty())
          {
            std::cout << "cvBGR.empty ..." << std::endl;
            for(int i=0; i < cam_count; i++)
            {
              ctx[i].enc->setEncoderCommand(V4L2_ENC_CMD_STOP, 1);
            }
            break;
          }

          cvBGR_queue[i].push(cvBGR[i]);
          /* Color transform from BGR to YUV_I420 */
          cv::cvtColor(cvBGR[i], cvYUV[i], cv::COLOR_BGR2YUV_I420);
        }        
      } //for_running                   
    }//for loop

    if (motion_vector_) 
    {      
      /* Using C++11 future async function to get multi-channel MVs */
      for(int i=0; i<cam_count; i++)
      {
        fuRes_enc[i] = std::async(std::launch::async, MvExtractor_Process, &ctx[i], &cvYUV[i]);
      }      
      
      int mv_ret = 0;
      for(int i=0; i<cam_count; i++)
      {     
        mv_ret += fuRes_enc[i].get();
      }
      
      if(mv_ret == 0)
      {
        for(int selected_channel=0; selected_channel<cam_count; selected_channel++)
        {
          SafeQueue<MotionVectors> *mv_queue = &(ctx[selected_channel].mvs_queue);

          if(mv_queue->size() > 1)
          {
            MotionVectors mv_struct;
            ctx[selected_channel].mvs_queue.pop(mv_struct);
            cvBGR_queue[selected_channel].pop(cvMV[selected_channel]);

            //std::cout << "get [" << mv_struct.frame_number << "] from queue, queue size = " << mv_queue->size() << ", elapsed = " << ctx[selected_channel].timestampincr/1000 << " ms" << std::endl;

            if(mv_struct.frame_number != debug_counter[selected_channel]++)
                    std::cerr << "MV not sync ... @" << mv_struct.frame_number << ", " << debug_counter[selected_channel] << std::endl;

            MVInfo *pInfo = (MVInfo *)mv_struct.mvs.data();
            msgs::MotionVector mv_msg;
            
            mv_msgs_array[selected_channel].mvs.clear(); //clear old elements

            for (uint32_t i = 0; i < mv_struct.total_numbers; i++, pInfo++)
            {
              int32_t dst_x = 16 * (i%(ctx[selected_channel].width/16)) + 8;
              int32_t dst_y = 16 * (i/(ctx[selected_channel].width/16)) + 8;
              int32_t src_x = dst_x + pInfo->mv_x/16;
              int32_t src_y = dst_y + pInfo->mv_y/16;
              
#ifdef MV_IMAGE_DEBUG
              cv::arrowedLine(cvMV[selected_channel], cv::Point(src_x, src_y), cv::Point(dst_x, dst_y), cv::Scalar(0, 255, 0), 2, 8, 0, 0.5);
#endif              
              mv_msg.mv_x = dst_x - src_x;
              mv_msg.mv_y = dst_y - src_y;
              mv_msg.src_x = src_x;
              mv_msg.src_y = src_y;
              
              mv_msgs_array[selected_channel].mvs.emplace_back(mv_msg);

            }
          }
        }//end for selected_channel
      }//end (mv_ret == 0)
     

    }//end motion_vector_
    
    check_green_screen = false; //only check green screen for first time

    for (i = 0; i < cam_count; i++)
    {
      if (for_running == false)
        break;

      ros_image.send_image_rgb_gstreamer(cam_ids_[i], canvas_tmp[i], ros_time_);
    
      
      if (motion_vector_) 
      {
#ifdef MV_IMAGE_DEBUG
        ros_image.send_image_rgb_gstreamer_mv(cam_ids_[i], cvMV[i], ros_time_);
#endif 
        ros_image.send_image_rgb_gstreamer_mv_msgs(cam_ids_[i], mv_msgs_array[i], ros_time_);
      }
    }

    loop_rate.sleep();

  }  // end while

  return true;
}
}  // namespace SensingSubSystem
