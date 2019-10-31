#include "TegraBGrabber.h"
#include "grabber_args_parser.h"

namespace SensingSubSystem
{
///
/// Class TegraBGrabber
///
TegraBGrabber::TegraBGrabber()
  : cam_ids_({camera::id::left_120, camera::id::front_120, camera::id::right_120})
  , canvas(cam_ids_.size())
  , grabber(nullptr)
  , npp8u_ptrs_(cam_ids_.size())
  , npp8u_ptrs_distorted_(cam_ids_.size())
  , resizer_(camera::raw_image_height, camera::raw_image_width, 384, 608)
  , remapper_(camera::raw_image_height, camera::raw_image_width)
  , ros_image(n)
{
  InitParameters();
}

void TegraBGrabber::InitParameters()
{
    int dummy;
    for (size_t i = 0; i < cam_ids_.size(); i++)
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
    for (int i = 0; i < 3; i++)
    {
        nppiFree(npp8u_ptrs_[i]);
        nppiFree(npp8u_ptrs_distorted_[i]);
    }
}

void TegraBGrabber::initializeModules()
{
    for (size_t i = 0; i < cam_ids_.size(); ++i)
    {
        const int cam_id = cam_ids_[i];
        ros_image.add_a_pub(cam_id, camera::topics[cam_id]);
    }
    
    grabber = new MultiGMSLCameraGrabber("000001110000");
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

    // start image processing
    for (size_t i =0 ; i < cam_ids_.size(); i++)
    {
        npp_wrapper::npp8u_ptr_c4_to_c3(static_cast<const Npp8u*>(
            camera_buffer_.cams_ptr->frames_GPU[cam_ids_[i]]), camera::raw_image_rows, camera::raw_image_cols, npp8u_ptrs_[i]);
        remapper_.remap(npp8u_ptrs_[i], npp8u_ptrs_distorted_[i]);
        resizer_.resize(npp8u_ptrs_distorted_[i], canvas[i]);
    }
    // end image processing

    // return camera grabber
    grabber->returnCameraFrame();

    // pub camera image through ros
    for (size_t i = 0; i < cam_ids_.size(); ++i)
    {
        ros_image.send_image_rgb(cam_ids_[i], canvas[i]);
    }

    loop_rate.sleep();

  }  // end while

  return true;
}
}
