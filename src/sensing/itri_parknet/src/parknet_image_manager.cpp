/*
   CREATER: ICL U300
   DATE: Feb, 2019
 */
#include <assert.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "camera_utils.h"
#include "camera_params.h"
#include "camera_distance_mapper.h"
#include "parknet_image_manager.h"
#include "parknet_camera.h"
#include "parknet_cv_colors.h"
#include "parknet_advertise_utils.h"
#include "npp.h"
#include "npp_utils.h"
#include "npp_remapper.h"
#include "npp_resizer.h"

ParknetImageManager::ParknetImageManager()
  : npp_remapper_ptr_(new NPPRemapper(camera::raw_image_height, camera::raw_image_width))
  , npp_resizer_ptr_(new NPPResizer(camera::raw_image_height, camera::raw_image_width, camera::yolov3_image_rows,
                                    camera::yolov3_image_cols))
{
  int dummy;
  for (int i = 0; i < parknet::camera::num_cams_e; i++)
  {
    npp8u_ptr_1920x1208_bgr_distorted_[i] = nppiMalloc_8u_C3(camera::raw_image_cols, camera::raw_image_rows, &dummy);
    npp8u_ptr_1920x1208_bgr_undistorted_[i] = nppiMalloc_8u_C3(camera::raw_image_cols, camera::raw_image_rows, &dummy);
    npp8u_ptr_608x608_bgr_undistorted_[i] =
        nppiMalloc_8u_C3(camera::yolov3_image_cols, camera::yolov3_image_rows, &dummy);
    npp8u_ptr_608x608_rgb_undistorted_[i] =
        nppiMalloc_8u_C3(camera::yolov3_image_cols, camera::yolov3_image_rows, &dummy);
    npp32f_ptr_608x608_rgb_undistorted_[i] =
        nppiMalloc_32f_C3(camera::yolov3_image_cols, camera::yolov3_image_rows, &dummy);
    npp32f_ptr_369664_rgb_[i] = nppiMalloc_32f_C1(camera::num_yolov3_image_pixels, 3, &dummy);

    CHECK(npp8u_ptr_1920x1208_bgr_distorted_[i]);
    CHECK(npp8u_ptr_1920x1208_bgr_undistorted_[i]);
    CHECK(npp8u_ptr_608x608_bgr_undistorted_[i]);
    CHECK(npp8u_ptr_608x608_rgb_undistorted_[i]);
    CHECK(npp32f_ptr_608x608_rgb_undistorted_[i]);
    CHECK(npp32f_ptr_369664_rgb_[i]);
  }
}

ParknetImageManager::~ParknetImageManager()
{
  for (int i = 0; i < parknet::camera::num_cams_e; i++)
  {
    release_image_if_necessary(raw_image_[i]);
    release_image_if_necessary(preprocessed_image_[i]);
    release_image_if_necessary(displayed_image_[i]);
    nppiFree(npp8u_ptr_1920x1208_bgr_distorted_[i]);
    nppiFree(npp8u_ptr_1920x1208_bgr_undistorted_[i]);
    nppiFree(npp8u_ptr_608x608_bgr_undistorted_[i]);
    nppiFree(npp8u_ptr_608x608_rgb_undistorted_[i]);
    nppiFree(npp32f_ptr_608x608_rgb_undistorted_[i]);
    nppiFree(npp32f_ptr_369664_rgb_[i]);
  }
}
void ParknetImageManager::release_image_if_necessary(cv::Mat& img_in)
{
  if (!img_in.empty())
  {
    img_in.release();
  }
}

int ParknetImageManager::get_raw_image(const int cam_id, cv::Mat& img_out)
{
  if (!img_out.empty())
  {
    img_out.release();
  }
  npp_wrapper::npp8u_ptr_to_cvmat(npp8u_ptr_1920x1208_bgr_distorted_[cam_id], camera::num_raw_image_bytes, img_out,
                                  camera::raw_image_rows, camera::raw_image_cols);
  return 0;
}

int ParknetImageManager::get_preprocessed_image(const int cam_id, cv::Mat& img_out)
{
  if (!img_out.empty())
  {
    img_out.release();
  }
  npp_wrapper::npp8u_ptr_to_cvmat(npp8u_ptr_1920x1208_bgr_undistorted_[cam_id], camera::num_raw_image_bytes, img_out,
                                  camera::raw_image_rows, camera::raw_image_cols);
  return 0;
}

/*
 * Getter function for image blob that is ready to feed into a neural network.
 **/
const Npp32f* ParknetImageManager::get_blob(const int cam_id)
{
  return npp32f_ptr_369664_rgb_[cam_id];
}

/*
 * Annotate the input image with bounding boxes. The output image size is 608x384.
 *
 * @param[in] cam_id Camera ID
 * @param[in] dist_mapper_ptr
 * @param[out] img_out
 **/
int ParknetImageManager::get_annotated_image(const int cam_id, const CameraDistanceMapper* dist_mapper_ptr,
                                             cv::Mat& img_out)
{
  if (!img_out.empty())
  {
    img_out.release();
  }
  {
    npp_wrapper::npp8u_ptr_to_cvmat(npp8u_ptr_608x608_bgr_undistorted_[cam_id], camera::num_yolov3_image_bytes, img_out,
                                    camera::yolov3_image_rows, camera::yolov3_image_cols);
  }
  gen_displayed_image_yolov3(cam_id, dist_mapper_ptr, img_out);
  return 0;
}

void ParknetImageManager::gen_displayed_image_yolov3(const int cam_id, const CameraDistanceMapper* dist_mapper_ptr,
                                                     cv::Mat& img_out)
{
  for (const auto& corner : detections_[cam_id])
  {
    drawpred_1(corner, dist_mapper_ptr, img_out, /*border width*/ 3);
  }
  // Remove top/botoom borders that are in black.
  cv::Rect roi(0, camera::top_border, camera::yolov3_image_width,
               camera::yolov3_image_height - camera::bottom_border - camera::top_border);
  cv::Mat temp = img_out(roi).clone();
  img_out.release();
  img_out = temp.clone();
}

void ParknetImageManager::drawpred_1(const RectClassScore<float>& corner, const CameraDistanceMapper* dist_mapper_ptr,
                                     cv::Mat& frame, int border_width)
{
  const int x = corner.x;
  const int y = corner.y;
  const int w = corner.w;
  const int h = corner.h;
  rectangle(frame, cv::Point(x, y), cv::Point(x + w, y + h), g_color_red, border_width);
  std::string label = corner.GetClassString() + cv::format(": %.2f", corner.score);

  VLOG(2) << "mark " << label << "(" << x << "," << y << "), (" << x + w << ", " << y + h << ") in image " << &frame;
  int baseLine;
  cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  int top = std::max(x, labelSize.height);
  putText(frame, label, cv::Point(top, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, g_color_white);

  // draw spatial x, y, z
  const auto marking_point = parknet::convert_corner_to_marking_point(corner, *dist_mapper_ptr);

  const auto spatial_x_label = cv::format("x: %.2fm", marking_point.x);
  putText(frame, spatial_x_label, cv::Point(x, y + h + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, g_color_red);
  const auto spatial_y_label = cv::format("y: %.2fm", marking_point.y);
  putText(frame, spatial_y_label, cv::Point(x, y + h + labelSize.height * 2), cv::FONT_HERSHEY_SIMPLEX, 0.5,
          g_color_red);
  const auto spatial_z_label = cv::format("z: %.2fm", marking_point.z);
  putText(frame, spatial_z_label, cv::Point(x, y + h + labelSize.height * 3), cv::FONT_HERSHEY_SIMPLEX, 0.5,
          g_color_red);
}

/*
 * Image process pipeline: Given a raw image of size 1920x1208, convert it into undistorted image and keep its blob.
 *
 * @param[in] npp8u_1920_1208_distorted
 * @param[in] cam_id
 **/
int ParknetImageManager::image_processing_pipeline(const Npp8u* npp8u_1920_1208_distorted, const int cam_id)
{
  if (npp8u_1920_1208_distorted != npp8u_ptr_1920x1208_bgr_distorted_[cam_id])
  {
    cudaMemcpyAsync(npp8u_ptr_1920x1208_bgr_distorted_[cam_id], npp8u_1920_1208_distorted, camera::num_raw_image_bytes,
                    cudaMemcpyHostToDevice, cudaStreamPerThread);
  }
  // undistort
  npp_remapper_ptr_->remap(npp8u_ptr_1920x1208_bgr_distorted_[cam_id], npp8u_ptr_1920x1208_bgr_undistorted_[cam_id]);

  // resize to yolov3 image size
  {
    std::lock_guard<std::mutex> lock(resizer_mutex_);
    npp_resizer_ptr_->resize_to_letterbox_yolov3(npp8u_ptr_1920x1208_bgr_undistorted_[cam_id],
                                                 npp8u_ptr_608x608_bgr_undistorted_[cam_id]);
  }

  // BRG -> RGB
  const int rgb_order[3] = { 2, 1, 0 };
  NppiSize roi = {.width = camera::yolov3_image_cols, .height = camera::yolov3_image_rows };
  auto status = nppiSwapChannels_8u_C3R(npp8u_ptr_608x608_bgr_undistorted_[cam_id], camera::num_yolov3_bytes_per_row_u8,
                                        npp8u_ptr_608x608_rgb_undistorted_[cam_id], camera::num_yolov3_bytes_per_row_u8,
                                        roi, rgb_order);
  assert(status == NPP_SUCCESS);
  NO_UNUSED_VAR_CHECK(status);

  // u8 -> 32f
  status =
      nppiConvert_8u32f_C3R(npp8u_ptr_608x608_rgb_undistorted_[cam_id], camera::num_yolov3_bytes_per_row_u8,
                            npp32f_ptr_608x608_rgb_undistorted_[cam_id], camera::num_yolov3_bytes_per_row_f32, roi);
  assert(status == NPP_SUCCESS);

  // flatten color channels
  npp_wrapper::blob_from_image(npp32f_ptr_608x608_rgb_undistorted_[cam_id], camera::yolov3_image_rows,
                               camera::yolov3_image_cols, npp32f_ptr_369664_rgb_[cam_id]);
  cudaDeviceSynchronize();  // Do not delete this line!
  return 0;
}

int ParknetImageManager::image_processing_pipeline_yolov3(const Npp8u* npp8u_608x608, const int cam_id)
{
  if (npp8u_608x608 != npp8u_ptr_608x608_bgr_undistorted_[cam_id])
  {
    cudaMemcpyAsync(npp8u_ptr_608x608_bgr_undistorted_[cam_id], npp8u_608x608, camera::num_yolov3_image_bytes,
                    cudaMemcpyHostToDevice, cudaStreamPerThread);
  }
  // BRG -> RGB
  const int rgb_order[3] = { 2, 1, 0 };
  NppiSize roi = {.width = camera::yolov3_image_cols, .height = camera::yolov3_image_rows };
  auto status = nppiSwapChannels_8u_C3R(npp8u_ptr_608x608_bgr_undistorted_[cam_id], camera::num_yolov3_bytes_per_row_u8,
                                        npp8u_ptr_608x608_rgb_undistorted_[cam_id], camera::num_yolov3_bytes_per_row_u8,
                                        roi, rgb_order);
  if (status != NPP_SUCCESS)
  {
    LOG(WARNING) << "nppiSwapChannels_8u_C3R returns: " << status;
    return 1;
  }
  assert(status == NPP_SUCCESS);

  // u8 -> 32f
  status =
      nppiConvert_8u32f_C3R(npp8u_ptr_608x608_rgb_undistorted_[cam_id], camera::num_yolov3_bytes_per_row_u8,
                            npp32f_ptr_608x608_rgb_undistorted_[cam_id], camera::num_yolov3_bytes_per_row_f32, roi);
  if (status != NPP_SUCCESS)
  {
    LOG(WARNING) << "nppiConvert_8u32f_C3R returns: " << status;
    return 1;
  }

  assert(status == NPP_SUCCESS);

  // flatten color channels
  npp_wrapper::blob_from_image(npp32f_ptr_608x608_rgb_undistorted_[cam_id], camera::yolov3_image_rows,
                               camera::yolov3_image_cols, npp32f_ptr_369664_rgb_[cam_id]);
  cudaDeviceSynchronize();  // Do not delete this line!
  return 0;
}

/*
 * Set raw image by Npp8u*.
 *
 * @param[in] npp8u_1920_1208_distorted
 * @param[in] cam_id
 * @param[in] num_bytes The length of |npp8u_1920_1208_distorted|.
 */
int ParknetImageManager::set_raw_image(const Npp8u* npp8u_ptr, const int cam_id, const int num_bytes)
{
  LOG_EVERY_N(INFO, 300) << "set_raw_image by Npp8u*";
  assert(npp8u_ptr);
  if (num_bytes == camera::num_raw_image_bytes)
  {
    return image_processing_pipeline(npp8u_ptr, cam_id);
  }
  else if (num_bytes == camera::num_yolov3_image_bytes)
  {
    return image_processing_pipeline_yolov3(npp8u_ptr, cam_id);
  }
  else
  {
    LOG(WARNING) << __FUNCTION__ << ": Not implement yet.";
    return 1;
  }
}

int ParknetImageManager::set_raw_image(const cv::Mat& img_in, const int cam_id)
{
  LOG_EVERY_N(INFO, 300) << "set_raw_image by cv::Mat " << img_in.cols << "x" << img_in.rows;
  if (camera::has_raw_image_size(img_in))
  {
    npp_wrapper::cvmat_to_npp8u_ptr(img_in, npp8u_ptr_1920x1208_bgr_distorted_[cam_id]);
    set_raw_image(npp8u_ptr_1920x1208_bgr_distorted_[cam_id], cam_id, camera::num_raw_image_bytes);
  }
  else if (camera::has_yolov3_image_size(img_in))
  {
    int dummy;
    Npp8u* npp8u_608x608 = nppiMalloc_8u_C3(camera::yolov3_image_cols, camera::yolov3_image_rows, &dummy);
    assert(npp8u_608x608);
    npp_wrapper::cvmat_to_npp8u_ptr(img_in, npp8u_608x608);
    set_raw_image(npp8u_608x608, cam_id, camera::num_yolov3_image_bytes);
    nppiFree(npp8u_608x608);
  }
  else if ((img_in.cols == 608) && (img_in.rows == 384))
  {
    cv::Mat img_yolov3;
    cv::copyMakeBorder(img_in, img_yolov3, camera::top_border_608x384, camera::bottom_border_608x384,
                       camera::left_border, camera::right_border, cv::BORDER_CONSTANT, g_color_black);
    assert(camera::has_yolov3_image_size(img_yolov3));
    npp_wrapper::cvmat_to_npp8u_ptr(img_yolov3, npp8u_ptr_608x608_bgr_undistorted_[cam_id]);
    set_raw_image(npp8u_ptr_608x608_bgr_undistorted_[cam_id], cam_id, camera::num_yolov3_image_bytes);
    img_yolov3.release();
  }
  else
  {
    LOG(WARNING) << "unsupport cv::Mat width x height: " << img_in.cols << "x" << img_in.rows;
    assert(0);
  }
  return 0;
}

int ParknetImageManager::set_detection(const std::vector<RectClassScore<float> >& in_detection, const int cam_id)
{
  detections_[cam_id].clear();
  detections_[cam_id].assign(in_detection.begin(), in_detection.end());
  return 0;
}

const std::vector<RectClassScore<float> > ParknetImageManager::get_detection(const int cam_id) const
{
  return detections_[cam_id];
}

int ParknetImageManager::get_num_detections(const int cam_id) const
{
  return detections_[cam_id].size();
}
