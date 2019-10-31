#include "TegraAGrabber.h"
#include "grabber_args_parser.h"
#include "camera_params.h"

namespace SensingSubSystem
{
///
/// Class TegraAGrabber
///
TegraAGrabber::TegraAGrabber()
  : cam_ids_({ camera::id::left_60, camera::id::front_60, camera::id::right_60, camera::id::left_30,
               camera::id::front_30, camera::id::right_30 })
  , canvas(cam_ids_.size())
  , display_(&camera_buffer_)
  , grabber(nullptr)
  , npp8u_ptrs_(cam_ids_.size())
  , resizer_(camera::raw_image_height, camera::raw_image_width, 384, 608)
  , ros_image(n)
{
  InitParameters();
}

void TegraAGrabber::InitParameters()
{
  for (size_t i = 0; i < cam_ids_.size(); i++)
  {
    int dummy;
    npp8u_ptrs_[i] = nppiMalloc_8u_C3(camera::raw_image_width, camera::raw_image_height, &dummy);
  }
}

TegraAGrabber::~TegraAGrabber()
{
  if (grabber != nullptr)
  {
    delete grabber;
    printf("grabber DELETED OK!\n");
  }
  for (int i = 0; i < 6; i++)
  {
    nppiFree(npp8u_ptrs_[i]);
  }
}

void TegraAGrabber::initializeModules()
{
  for (size_t i = 0; i < cam_ids_.size(); ++i)
  {
    const int cam_id = cam_ids_[i];
    ros_image.add_a_pub(cam_id, camera::topics[cam_id]);
  }

  grabber = new MultiGMSLCameraGrabber("011100000111");
  grabber->initializeCameras();
  camera_buffer_.initBuffer();

  printf("init done!\n");
}

bool TegraAGrabber::runPerception()
{
  auto fps = SensingSubSystem::get_expected_fps();
  ros::Rate loop_rate(fps);

  while (ros::ok())
  {
    // update vehicle state & imu state
    ros::spinOnce();

    // Using default stream for retreiving latest image
    grabber->retrieveNextFrame();

    cudaMemcpy(camera_buffer_.cams_ptr->frames_GPU[0], grabber->getCurrentFrameData(0),
               MultiGMSLCameraGrabber::ImageSize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(camera_buffer_.cams_ptr->frames_GPU[1], grabber->getCurrentFrameData(1),
               MultiGMSLCameraGrabber::ImageSize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(camera_buffer_.cams_ptr->frames_GPU[2], grabber->getCurrentFrameData(2),
               MultiGMSLCameraGrabber::ImageSize, cudaMemcpyDeviceToDevice);

    cudaMemcpy(camera_buffer_.cams_ptr->frames_GPU[8], grabber->getCurrentFrameData(8),
               MultiGMSLCameraGrabber::ImageSize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(camera_buffer_.cams_ptr->frames_GPU[9], grabber->getCurrentFrameData(9),
               MultiGMSLCameraGrabber::ImageSize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(camera_buffer_.cams_ptr->frames_GPU[10], grabber->getCurrentFrameData(10),
               MultiGMSLCameraGrabber::ImageSize, cudaMemcpyDeviceToDevice);

    // start image processing

    npp_wrapper::npp8u_ptr_c4_to_c3(static_cast<const Npp8u*>(camera_buffer_.cams_ptr->frames_GPU[0]),
                                    camera::raw_image_rows, camera::raw_image_cols, npp8u_ptrs_[0]);
    resizer_.resize(npp8u_ptrs_[0], canvas[0]);

    npp_wrapper::npp8u_ptr_c4_to_c3(static_cast<const Npp8u*>(camera_buffer_.cams_ptr->frames_GPU[1]),
                                    camera::raw_image_rows, camera::raw_image_cols, npp8u_ptrs_[1]);
    resizer_.resize(npp8u_ptrs_[1], canvas[1]);

    npp_wrapper::npp8u_ptr_c4_to_c3(static_cast<const Npp8u*>(camera_buffer_.cams_ptr->frames_GPU[2]),
                                    camera::raw_image_rows, camera::raw_image_cols, npp8u_ptrs_[2]);
    resizer_.resize(npp8u_ptrs_[2], canvas[2]);

    npp_wrapper::npp8u_ptr_c4_to_c3(static_cast<const Npp8u*>(camera_buffer_.cams_ptr->frames_GPU[8]),
                                    camera::raw_image_rows, camera::raw_image_cols, npp8u_ptrs_[3]);
    resizer_.resize(npp8u_ptrs_[3], canvas[3]);

    npp_wrapper::npp8u_ptr_c4_to_c3(static_cast<const Npp8u*>(camera_buffer_.cams_ptr->frames_GPU[9]),
                                    camera::raw_image_rows, camera::raw_image_cols, npp8u_ptrs_[4]);
    resizer_.resize(npp8u_ptrs_[4], canvas[4]);

    npp_wrapper::npp8u_ptr_c4_to_c3(static_cast<const Npp8u*>(camera_buffer_.cams_ptr->frames_GPU[10]),
                                    camera::raw_image_rows, camera::raw_image_cols, npp8u_ptrs_[5]);
    resizer_.resize(npp8u_ptrs_[5], canvas[5]);

    // end image processing

    // return camera grabber
    grabber->returnCameraFrame();

    for (size_t i = 0; i < cam_ids_.size(); ++i)
    {
      ros_image.send_image_rgb(cam_ids_[i], canvas[i]);
    }

    loop_rate.sleep();

  }  // end while

  return true;
}
}
