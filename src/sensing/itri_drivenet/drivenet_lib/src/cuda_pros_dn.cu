#include <stdio.h>

#include <cuda_runtime.h>
#include "cuda_pros_dn.h"

dim3 block;
dim3 grid;
int thread_blocks;
float x_ratio;
float y_ratio;

__global__ void cuda_preprocess_for_cv(unsigned char *d_out, int result_width, int result_height, unsigned char *d_in, int ori_width, int ori_height, int N, float x_ratio, float y_ratio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        int result_y = idx / result_width;
        int result_x = idx % result_width;

        int ori_y = result_y * y_ratio;
        int ori_x = result_x * x_ratio;

        int ori_idx = ori_y * ori_width + ori_x;
        
        d_out[2*N+idx] = d_in[ori_idx*4];
        d_out[N+idx] = d_in[ori_idx*4+1];
        d_out[idx] = d_in[ori_idx*4+2];
        
        //printf("idx: %d result_y: %d  result_x: %d ori_y: %d ori_x: %d x_ratio: %f y_ratio: %f ori_idx: %d\n", idx, result_y, result_x, ori_y, ori_x, x_ratio, y_ratio, ori_idx);
    }
}

extern "C" void cudaResize_gpu_memory_preprocess_for_cv(unsigned char *d_out, int result_width, int result_height, unsigned char *d_in, int ori_width, int ori_height, int N)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    block.x = 512;
    thread_blocks = N / block.x;
    grid.x = thread_blocks % 65535 * 8;
    grid.y = (thread_blocks / 65535 + 1);

    x_ratio = (float)ori_width / (float)result_width; 
    y_ratio = (float)ori_height / (float)result_height;  

    cuda_preprocess_for_cv <<< grid, block >>> (d_out, result_width, result_height, d_in, ori_width, ori_height, N, x_ratio, y_ratio);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //printf("elapsedTime: %f ms\n", elapsedTime);
}

__global__ void cuda_preprocess_gpu_memory(float *d_out, int result_width, int result_height, unsigned char *d_in, int ori_width, int ori_height, int N, float x_ratio, float y_ratio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        int result_y = idx / result_width;
        int result_x = idx % result_width;

        int ori_y = result_y * y_ratio;
        int ori_x = result_x * x_ratio;

        int ori_idx = ori_y * ori_width + ori_x;
        
        d_out[2*N+idx] = (float)d_in[ori_idx*4];
        d_out[N+idx] = (float)d_in[ori_idx*4+1];
        d_out[idx] = (float)d_in[ori_idx*4+2];
        
        //printf("idx: %d result_y: %d  result_x: %d ori_y: %d ori_x: %d x_ratio: %f y_ratio: %f ori_idx: %d\n", idx, result_y, result_x, ori_y, ori_x, x_ratio, y_ratio, ori_idx);
    }
}

extern "C" void cudaResize_gpu_memory_preprocess(float *d_out, int result_width, int result_height, unsigned char *d_in, int ori_width, int ori_height, int N)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    block.x = 512;
    thread_blocks = N / block.x;
    grid.x = thread_blocks % 65535 * 8;
    grid.y = (thread_blocks / 65535 + 1);

    x_ratio = (float)ori_width / (float)result_width; 
    y_ratio = (float)ori_height / (float)result_height;  

    cuda_preprocess_gpu_memory <<< grid, block >>> (d_out, result_width, result_height, d_in, ori_width, ori_height, N, x_ratio, y_ratio);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //printf("elapsedTime: %f ms\n", elapsedTime);
}

__global__ void cuda_preprocess_video(float *d_out, int result_width, int result_height, float *d_in, int ori_width, int ori_height, int N, float x_ratio, float y_ratio)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        int result_y = idx / result_width;
        int result_x = idx % result_width;

        int ori_y = result_y * y_ratio;
        int ori_x = result_x * x_ratio;

        int ori_idx = ori_y * ori_width + ori_x;

        /*
        d_out[idx*3] = d_in[ori_idx*3];
        d_out[idx*3+1] = d_in[ori_idx*3+1];
        d_out[idx*3+2] = d_in[ori_idx*3+2];
        */
        
        d_out[idx] = d_in[ori_idx*3] - 103.939f;
        d_out[N+idx] = d_in[ori_idx*3+1] - 116.779f;
        d_out[2*N+idx] = d_in[ori_idx*3+2] - 123.68f;
        
        //printf("idx: %d result_y: %d  result_x: %d ori_y: %d ori_x: %d x_ratio: %f y_ratio: %f ori_idx: %d\n", idx, result_y, result_x, ori_y, ori_x, x_ratio, y_ratio, ori_idx);
    }
}

extern "C" void cudaResize_video(float *d_out, int result_width, int result_height, float *d_in, int ori_width, int ori_height, int N)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    block.x = 512;
    thread_blocks = N / block.x;
    grid.x = thread_blocks % 65535 * 8;
    grid.y = (thread_blocks / 65535 + 1);

    x_ratio = (float)ori_width / (float)result_width; 
    y_ratio = (float)ori_height / (float)result_height;  

    cuda_preprocess_video <<< grid, block >>> (d_out, result_width, result_height, d_in, ori_width, ori_height, N, x_ratio, y_ratio);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //printf("elapsedTime: %f ms\n", elapsedTime);
}

__global__ void incKernel(float *g_out, float *g_in, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) // know what time end ???????????????????????????????
    {
        //printf("idx: %d\n", idx);
        g_out[idx] = g_in[idx * 3 + 0];          // out 0 in 0
        g_out[N + idx] = g_in[idx * 3 + 1];      // out 512*256 in 1
        g_out[2 * N + idx] = g_in[idx * 3 + 2];  // out 2*512*256 in 2
    }
}

extern "C" void cudaReshape(float *g_out, float *g_in, int N)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    block.x = 512;
    thread_blocks = N / block.x;
    grid.x = thread_blocks % 65535 * 8;
    grid.y = (thread_blocks / 65535 + 1);

    incKernel<<<grid, block>>>(g_out, g_in, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //printf("elapsedTime: %f ms\n", elapsedTime);
}
