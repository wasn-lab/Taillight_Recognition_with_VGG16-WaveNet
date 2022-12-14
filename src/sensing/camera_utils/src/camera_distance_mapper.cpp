#include <fstream>
#include <cassert>
#include <jsoncpp/json/json.h>
#include <glog/logging.h>
#include <opencv2/imgproc.hpp>
#include "car_model.h"
#include "camera_utils_defs.h"
#include "camera_utils.h"
#include "camera_distance_mapper.h"

CameraDistanceMapper::CameraDistanceMapper(const camera::id cam_id)
  : cam_id_(cam_id)
  , x_in_meters_ptr_(new float[camera::raw_image_rows][camera::raw_image_cols])
  , y_in_meters_ptr_(new float[camera::raw_image_rows][camera::raw_image_cols])
  , z_in_meters_ptr_(new float[camera::raw_image_rows][camera::raw_image_cols])
{
  assert(cam_id >= camera::id::begin);
  assert(cam_id <= camera::id::num_ids);
#if CAR_MODEL_IS_HINO
  // Currently only FOV120 are supported
  assert(cam_id >= camera::id::top_front_120);
  assert(cam_id <= camera::id::top_rear_120);
#endif

  read_dist_from_json();
}

int CameraDistanceMapper::read_dist_from_json()
{
  std::string jfile = get_json_filename();
  LOG(INFO) << "read " << jfile << " to get spatial distance.";

  std::ifstream ifs(jfile);
  Json::Reader jreader;
  Json::Value jdata;
  jreader.parse(ifs, jdata);

  for (const auto& doc : jdata)
  {
    auto image_x = doc["im_x"].asInt();
    auto image_y = doc["im_y"].asInt();
    assert(image_x >= 0);
    assert(image_y >= 0);

    if ((image_y < camera::raw_image_rows) && (image_x < camera::raw_image_cols))
    {
      x_in_meters_ptr_[image_y][image_x] = doc["dist_in_cm"][0].asInt() / 100.0;
      y_in_meters_ptr_[image_y][image_x] = doc["dist_in_cm"][1].asInt() / 100.0;
      z_in_meters_ptr_[image_y][image_x] = doc["dist_in_cm"][2].asInt() / 100.0;
    }
  }
  return 0;
}

// Experimental: Use it at your own risk
cv::Mat CameraDistanceMapper::remap_distance_in_undistorted_image()
{
  cv::Mat mapx, mapy;
  cv::Mat res, temp;

  camera::get_undistortion_maps(mapx, mapy);
  temp.create(camera::raw_image_rows, camera::raw_image_cols, CV_32FC1);
  memcpy(temp.data, x_in_meters_ptr_.get(), sizeof(float) * 1920 * 1208);
  cv::remap(temp, res, mapx, mapy, cv::INTER_LINEAR);
  return res;
}

/*
 * Map the image pixel at (im_x, im_y) into spatial distance (spatial_x, spatial_y, spatial_z)
 * This function is for raw image (distorted 1920x1208 image in the FOV120 case).
 *
 * @param[in] im_x
 * @param[in] im_y
 * @parma[out] *spatial_x
 * @parma[out] *spatial_y
 * @parma[out] *spatial_z
 **/
int CameraDistanceMapper::get_distance_raw_1920x1208(const int im_x, const int im_y, float* spatial_x, float* spatial_y,
                                                     float* spatial_z) const
{
  assert(im_x > 0);
  assert(im_y > 0);
  assert(im_x < camera::raw_image_width);
  assert(im_y < camera::raw_image_height);
  *spatial_x = x_in_meters_ptr_[im_y][im_x];
  *spatial_y = y_in_meters_ptr_[im_y][im_x];
  *spatial_z = z_in_meters_ptr_[im_y][im_x];
  return 0;
}

std::string CameraDistanceMapper::get_json_filename()
{
  std::string data_dir = CAMERA_UTILS_DATA_DIR;
  std::string json_filenames[] = {
    data_dir + "/",  // 0: dummy
    data_dir + "/",  // 1: left_60
    data_dir + "/",
    data_dir + "/",
    data_dir + "/left_120.json",   // 4: left_120
    data_dir + "/front_120.json",  // 5: front_120
    data_dir + "/right_120.json",  // 6: right_120
    data_dir + "/",
    data_dir + "/",
    data_dir + "/",
  };
  return json_filenames[cam_id_];
}
