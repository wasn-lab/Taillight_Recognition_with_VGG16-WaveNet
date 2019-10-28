#include "TegraAGrabber.h"
#include "grabber_args_parser.h"

namespace SensingSubSystem
{
///
/// Class TegraAGrabber
///
TegraAGrabber::TegraAGrabber()
  : display_(&camera_buffer_), grabber(nullptr)
  , npp8u_ptrs_(6), ros_image(n), canvas(6)
  , resizer_(camera::raw_image_height, camera::raw_image_width, 384, 608)
{
  InitParameters();
}

void TegraAGrabber::InitParameters()
{
  std::vector<size_t> _image_num{ 0, 1, 2, 8, 9, 10 };

  image_num = _image_num;

  for (int i = 0; i < 6; i++)
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
  for (size_t i = 0; i < image_num.size(); ++i)
  {
    ros_image.add_a_pub(image_num[i], "gmsl_camera/" + std::to_string(image_num[i]));
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

    for (size_t i = 0; i < image_num.size(); ++i)
    {
      ros_image.send_image_rgb(image_num[i], canvas[i]);
    }

    loop_rate.sleep();

  }  // end while

  return true;
}
}
