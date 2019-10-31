#include "TegraBGrabber.h"
#include "grabber_args_parser.h"

namespace SensingSubSystem
{
///
/// Class TegraBGrabber
///
TegraBGrabber::TegraBGrabber()
  : grabber(nullptr), npp8u_ptrs_distorted_(6)
  , npp8u_ptrs_(6), ros_image(n), canvas(6)
  , remapper_(camera::raw_image_height, camera::raw_image_width)
  , resizer_(camera::raw_image_height, camera::raw_image_width, 384, 608)
{
  InitParameters();
}

void TegraBGrabber::InitParameters()
{
    std::vector<size_t> _image_num{ 4, 5, 6 , 8, 9, 10};
    
    image_num = _image_num;
    
    int dummy;
    for (int i = 0; i < 6; i++)
    {
        npp8u_ptrs_[i] = nppiMalloc_8u_C3(camera::raw_image_width, camera::raw_image_height, &dummy);
        npp8u_ptrs_distorted_[i] = nppiMalloc_8u_C3(camera::raw_image_cols, camera::raw_image_rows, &dummy);
    }
}

TegraBGrabber::~TegraBGrabber()
{
    if (grabber != nullptr)
    {
        delete grabber;
        printf("grabber DELETED OK!\n");
    }
    for (int i = 0; i < 6; i++)
    {
        nppiFree(npp8u_ptrs_[i]);
        nppiFree(npp8u_ptrs_distorted_[i]);
    }
}

void TegraBGrabber::initializeModules()
{
    for (size_t i = 0; i < image_num.size(); ++i)
    {
        ros_image.add_a_pub(image_num[i], "gmsl_camera/" + std::to_string(image_num[i]));
    }
    
    grabber = new MultiGMSLCameraGrabber("000001110111");
    grabber->initializeCameras();
    camera_buffer_.initBuffer();
    
    printf("init done!\n");
}

bool TegraBGrabber::runPerception()
{
  
  auto fps = SensingSubSystem::get_expected_fps();

  ros::Rate loop_rate(fps);
  while (ros::ok())
  {
    // update vehicle state & imu state
    ros::spinOnce();

    // Using default stream for retreiving latest image
    grabber->retrieveNextFrame();
    cudaMemcpy(camera_buffer_.cams_ptr->frames_GPU[4], grabber->getCurrentFrameData(4),
               MultiGMSLCameraGrabber::ImageSize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(camera_buffer_.cams_ptr->frames_GPU[5], grabber->getCurrentFrameData(5),
               MultiGMSLCameraGrabber::ImageSize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(camera_buffer_.cams_ptr->frames_GPU[6], grabber->getCurrentFrameData(6),
               MultiGMSLCameraGrabber::ImageSize, cudaMemcpyDeviceToDevice);

    cudaMemcpy(camera_buffer_.cams_ptr->frames_GPU[8], grabber->getCurrentFrameData(8),
               MultiGMSLCameraGrabber::ImageSize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(camera_buffer_.cams_ptr->frames_GPU[9], grabber->getCurrentFrameData(9),
               MultiGMSLCameraGrabber::ImageSize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(camera_buffer_.cams_ptr->frames_GPU[10], grabber->getCurrentFrameData(10),
               MultiGMSLCameraGrabber::ImageSize, cudaMemcpyDeviceToDevice);

    // start image processing
    for (int i =0 ; i < image_num.size(); i++)
    {
        npp_wrapper::npp8u_ptr_c4_to_c3(static_cast<const Npp8u*>(
            camera_buffer_.cams_ptr->frames_GPU[image_num[i]]), camera::raw_image_rows, camera::raw_image_cols, npp8u_ptrs_[i]);
        remapper_.remap(npp8u_ptrs_[i], npp8u_ptrs_distorted_[i]);
        resizer_.resize(npp8u_ptrs_distorted_[i], canvas[i]);
    }
    // end image processing

    // return camera grabber
    grabber->returnCameraFrame();

    // pub camera image through ros
    for (size_t i = 0; i < image_num.size(); ++i)
    {
        ros_image.send_image_rgb(image_num[i], canvas[i]);
    }

    loop_rate.sleep();

  }  // end while

  return true;
}
}
