#ifndef _YOLO_CONFIG_PARSER_
#define _YOLO_CONFIG_PARSER_

#include "yolo.h"

#include <ctime>
#include <gflags/gflags.h>
namespace DriveNet
{
// Init to be called at the very beginning to verify all config params are valid
void yoloConfigParserInit(int argc, char** argv, std::string pkg_path);

NetworkInfo getYoloNetworkInfo();
InferParams getYoloInferParams();
uint64_t getSeed();
std::string getNetworkType();
std::string getPrecision();
std::string getTestImages();
std::string getTestImagesPath();
bool getDecode();
bool getDoBenchmark();
bool getViewDetections();
bool getSaveDetections();
std::string getSaveDetectionsPath();
uint getBatchSize();
bool getShuffleTestSet();
} // namespace DriveNet
#endif  //_YOLO_CONFIG_PARSER_
