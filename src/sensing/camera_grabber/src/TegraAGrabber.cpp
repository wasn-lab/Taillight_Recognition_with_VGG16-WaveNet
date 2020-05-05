#include "TegraAGrabber.h"
#include "grabber_args_parser.h"
#include "camera_params.h"

namespace SensingSubSystem
{
///
/// Class TegraAGrabber
///
TegraAGrabber::TegraAGrabber()
  : canvas(cam_ids_.size())
  , display_(&camera_buffer_)
  , grabber(nullptr)
  , npp8u_ptrs_(cam_ids_.size())
  , resizer_(camera::raw_image_height, camera::raw_image_width, camera::image_height,
             camera::image_width)  // 1208,1920 to 384,608
  , num_src_bytes_(camera::raw_image_height * camera::raw_image_width * 3)
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
    std::cout << "grabber DELETED OK!\n" << std::endl;
  }
  for (size_t i = 0; i < cam_ids_.size(); i++)
  {
    nppiFree(npp8u_ptrs_[i]);
  }
}

void TegraAGrabber::initializeModules(const bool do_resize, const bool do_crop)
{
  for (const auto cam_id : cam_ids_)
  {
    ros_image.add_a_pub(cam_id, camera::topics[cam_id]);
  }

  grabber = new MultiGMSLCameraGrabber("001100000000");
  grabber->initializeCameras();
  camera_buffer_.initBuffer();
  resize_ = do_resize;
  crop_ = do_crop;

  std::cout << "init done!\n" << std::endl;
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
    cudaMemcpy(camera_buffer_.cams_ptr->frames_GPU[2], grabber->getCurrentFrameData(0),
               MultiGMSLCameraGrabber::ImageSize, cudaMemcpyDeviceToDevice);

    // start image processing
    npp_wrapper::npp8u_ptr_c4_to_c3(static_cast<const Npp8u*>(camera_buffer_.cams_ptr->frames_GPU[0]),
                                    camera::raw_image_rows, camera::raw_image_cols, npp8u_ptrs_[0]);
    if (resize_)
    {
      resizer_.resize(npp8u_ptrs_[0], canvas[0]);
    }
    else
    {
      npp_wrapper::npp8u_ptr_to_cvmat(npp8u_ptrs_[0], num_src_bytes_, canvas[0], camera::raw_image_height,
                                      camera::raw_image_width);
    }

    npp_wrapper::npp8u_ptr_c4_to_c3(static_cast<const Npp8u*>(camera_buffer_.cams_ptr->frames_GPU[1]),
                                    camera::raw_image_rows, camera::raw_image_cols, npp8u_ptrs_[1]);
    if (resize_)
    {
      resizer_.resize(npp8u_ptrs_[1], canvas[1]);
    }
    else
    {
      npp_wrapper::npp8u_ptr_to_cvmat(npp8u_ptrs_[1], num_src_bytes_, canvas[1], camera::raw_image_height,
                                      camera::raw_image_width);
    }

     npp_wrapper::npp8u_ptr_c4_to_c3(static_cast<const Npp8u*>(camera_buffer_.cams_ptr->frames_GPU[2]),
                                     camera::raw_image_rows, camera::raw_image_cols, npp8u_ptrs_[2]);
     if (crop_){
         //resizer_.resize(npp8u_ptrs_[2], canvas[2]);
int dummy;
Npp8u* aDst = nppiMalloc_8u_C3(camera::image_crop_width, camera::image_crop_height, &dummy);

Npp8u* const aSrc = npp8u_ptrs_[2] + camera::raw_image_cols * 3*camera::image_crop_ystart+camera::image_crop_xstart ;
	

  	nppiCopy_8u_C3R(aSrc,  camera::raw_image_cols * 3, aDst,  camera::raw_image_cols * 3,
                                   {.width = camera::image_crop_width, .height = camera::image_crop_height });

npp_wrapper::npp8u_ptr_to_cvmat(aDst, camera::image_crop_height * camera::image_crop_width * 3, canvas[2], camera::image_crop_height,
                                      camera::image_crop_width);
nppiFree(aDst);
     }
/*
else
    {
      npp_wrapper::npp8u_ptr_to_cvmat(npp8u_ptrs_[2], num_src_bytes_, canvas[2], camera::raw_image_height,
                                      camera::raw_image_width);
    }
*/

    // end image processing

    // return camera grabber
    grabber->returnCameraFrame();
    int cam_count = cam_ids_.size();
    if(!crop_)
    {
      cam_count=cam_ids_.size()-1;
    }
    for (size_t i = 0; i < cam_count; ++i)
    {
      ros_image.send_image_rgb(cam_ids_[i], canvas[i]);
    }

    loop_rate.sleep();

  }  // end while

  return true;
}
}  // namespace SensingSubSystem
