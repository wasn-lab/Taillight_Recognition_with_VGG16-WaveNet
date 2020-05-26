#include "deeplab_segmenter.h"
#include "deeplab_const.h"
#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <glog/logging.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <fstream>

int main(int argc, char* argv[])
{
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InstallFailureSignalHandler();
  cv::setNumThreads(0);  // Avoid excessive CPU load
  deeplab::DeeplabSegmenter segmenter;

  std::string filename;
  std::unique_ptr<uint8_t[]> labels{ new uint8_t[deeplab::NUM_PIXELS] };
  while (std::getline(std::cin, filename))
  {
    if (access(filename.c_str(), F_OK) == -1)
    {
      LOG(WARNING) << "File not exist: " << filename;
    }
    std::string output_filename = filename.substr(0, filename.size() - 4) + "_deeplab.png";
    auto input = cv::imread(filename);
    cv::Mat output(cv::Size(deeplab::DEEPLAB_IMAGE_WIDTH, deeplab::DEEPLAB_IMAGE_HEIGHT), CV_8UC1);
    segmenter.segment_into_labels(input, labels.get());
    memcpy(output.data, labels.get(), sizeof(uint8_t) * deeplab::NUM_PIXELS);
    LOG(INFO) << "Write " << output_filename;
    cv::imwrite(output_filename, output);
  }

  return 0;
}
