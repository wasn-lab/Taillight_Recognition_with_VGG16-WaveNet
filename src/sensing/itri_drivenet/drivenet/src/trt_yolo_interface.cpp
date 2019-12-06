#include "drivenet/trt_yolo_interface.h"

using namespace DriveNet;
void Yolo_app::init_yolo(std::string pkg_path, std::string cfg_file)
{
  int argc = 2;
  std::string cfgFile("--flagfile=");
  cfgFile += pkg_path + cfg_file;
  std::cout << cfgFile.c_str() << std::endl;
  char* argv[2] = { (char*)"trt-yolo-app", (char*)cfgFile.c_str() };

  yoloConfigParserInit(argc, argv, pkg_path);
  NetworkInfo yoloInfo = getYoloNetworkInfo();
  InferParams yoloInferParams = getYoloInferParams();
  uint64_t seed = getSeed();
  std::string networkType = getNetworkType();
  std::string precision = getPrecision();
  std::string testImages = getTestImages();
  std::string testImagesPath = getTestImagesPath();

  decode = getDecode();
  doBenchmark = getDoBenchmark();
  viewDetections = getViewDetections();
  saveDetections = getSaveDetections();
  std::string saveDetectionsPath = getSaveDetectionsPath();
  batchSize = getBatchSize();
  shuffleTestSet = getShuffleTestSet();

  srand(unsigned(seed));
  inferYolo = new YoloV3(batchSize, yoloInfo, yoloInferParams);
  dsImags.init(inferYolo->getInputW(), inferYolo->getInputH());
  cudaMalloc((void**)&yoloInput, batchSize * inferYolo->getInputH() * inferYolo->getInputW() * 3 * sizeof(float));
}

void Yolo_app::input_preprocess(std::vector<cv::Mat*>& matSrcs)
{
  dsImgs_rows.clear();
  dsImgs_cols.clear();
  for (uint i = 0; i < batchSize; i++)
  {
    int img_h = (*matSrcs[i]).rows;
    int img_w = (*matSrcs[i]).cols;
    float* tmp = dsImags.preprocessing(*matSrcs[i], inferYolo->getInputH(), inferYolo->getInputW());
    dsImgs_rows.push_back(img_h);
    dsImgs_cols.push_back(img_w);

    cudaMemcpyAsync(yoloInput, tmp, inferYolo->getInputH() * inferYolo->getInputW() * 3 * sizeof(float),
                    cudaMemcpyDeviceToDevice, inferYolo->m_CudaStream);
    cudaFree(tmp);
    yoloInput += inferYolo->getInputH() * inferYolo->getInputW() * 3;
  }
  cudaStreamSynchronize(inferYolo->m_CudaStream);
  yoloInput = yoloInput - inferYolo->getInputH() * inferYolo->getInputW() * 3 * batchSize;
}
void Yolo_app::input_preprocess(std::vector<cv::Mat*>& matSrcs, int input_size, std::vector<int> dist_cols,
                                std::vector<int> dist_rows)
{
  dsImgs_rows.clear();
  dsImgs_cols.clear();
  dsImgs_cols = dist_cols;
  dsImgs_rows = dist_rows;

  for (uint i = 0; i < batchSize; i++)
  {
    float* tmp = dsImags.preprocessing(*matSrcs[i], inferYolo->getInputH(), inferYolo->getInputW(), input_size);

    cudaMemcpyAsync(yoloInput, tmp, inferYolo->getInputH() * inferYolo->getInputW() * 3 * sizeof(float),
                    cudaMemcpyDeviceToDevice, inferYolo->m_CudaStream);
    cudaFree(tmp);
    yoloInput += inferYolo->getInputH() * inferYolo->getInputW() * 3;
  }
  cudaStreamSynchronize(inferYolo->m_CudaStream);
  yoloInput = yoloInput - inferYolo->getInputH() * inferYolo->getInputW() * 3 * batchSize;
}
void Yolo_app::inference_yolo()
{
  inferYolo->doInference(yoloInput, dsImgs_cols.size());
}

void Yolo_app::get_yolo_result(std::vector<uint32_t>* order, std::vector<std::vector<ITRI_Bbox>*>& vbbx_output)
{
  for (uint imageIdx = 0; imageIdx < dsImgs_cols.size(); ++imageIdx)
  {
    auto binfo = inferYolo->decodeDetections(imageIdx, dsImgs_rows.at(imageIdx), dsImgs_cols.at(imageIdx));
    auto remaining = nmsAllClasses(inferYolo->getNMSThresh(), binfo, inferYolo->getNumClasses());

    std::vector<ITRI_Bbox>* tmpOut = vbbx_output[imageIdx];
    for (auto b : remaining)
    {
      ITRI_Bbox tmpBBx;
      tmpBBx.label = b.label;
      tmpBBx.classId = b.classId;
      tmpBBx.prob = b.prob;
      tmpBBx.x1 = b.box.x1;
      tmpBBx.y1 = b.box.y1;
      tmpBBx.x2 = b.box.x2;
      tmpBBx.y2 = b.box.y2;
      tmpOut->push_back(tmpBBx);
    }
  }
}

void Yolo_app::delete_yolo_infer()
{
  delete inferYolo;
}
