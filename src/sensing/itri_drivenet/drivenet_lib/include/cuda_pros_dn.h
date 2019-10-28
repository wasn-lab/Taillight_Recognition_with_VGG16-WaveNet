#ifndef CUDA_PROS_H
#define CUDA_PROS_H
namespace DriveNet{

extern "C" void cudaReshape(float *g_out, float *g_in, int N);
extern "C" void cudaResize_video(float *d_out, int result_width, int result_height, float *d_in, int ori_width, int ori_height, int N);
extern "C" void cudaResize_gpu_memory_preprocess(float *d_out, int result_width, int result_height, unsigned char *d_in, int ori_width, int ori_height, int N);
extern "C" void cudaResize_gpu_memory_preprocess_for_cv(unsigned char *d_out, int result_width, int result_height, unsigned char *d_in, int ori_width, int ori_height, int N);
}
#endif
