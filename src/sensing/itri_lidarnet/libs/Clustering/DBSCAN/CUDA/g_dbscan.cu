#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"

#include <thrust/device_vector.h>

__global__ void
_cu_vertdegree(int numpts, int colsize, float* eps, float* d_data, int* d_Va, int label_mode)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= numpts)
    return;

  d_Va[i] = 0;

  for (int j = 0; j < numpts; ++j) {
    float accum = 0.0;
    int colxyzsize = 3;
    int label = d_data[i * colsize + 4]
    
    for (int cs = 0; cs < colxyzsize; ++cs) {
      accum += (d_data[i * colsize + cs] - d_data[j * colsize + cs]) * (d_data[i * colsize + cs] - d_data[j * colsize + cs]);
    }
    
    if (label_mode) {
      if (accum < (eps[label]*eps[label]) && label == d_data[j * colsize + 4]) {
        d_Va[i] += 1;
      }
    }
    else
    {
      if (accum < (eps[0]*eps[0])) {
        d_Va[i] += 1;
      }
    }
  }
}

void vertdegree(int N, int colsize, float* eps, float* d_data, int* d_Va, int maxThreadsNumber, int label_mode)
{
  _cu_vertdegree<<<(N + 255) / 256, 256>>>(N, colsize, eps, d_data, d_Va, label_mode);
  //cudaDeviceSynchronize();
}

void adjlistsind(int N, int* Va0, int* Va1)
{
  thrust::device_ptr<int> va0_ptr(Va0);
  thrust::device_ptr<int> va1_ptr(Va1);
  thrust::exclusive_scan(va0_ptr, va0_ptr+N, va1_ptr);
  //cudaDeviceSynchronize();
}

__global__ void 
_cu_asmadjlist(int numpts, int colsize, float* eps, float* d_data, int* d_Va1, int* d_Ea, int label_mode)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= numpts)
    return;

  int basei = d_Va1[i];

  for (int j = 0; j < numpts; ++j) {
    float accum = 0.0;
    int colxyzsize = 3;
    int label = d_data[i * colsize + 4]
    
    for (int cs = 0; cs < colxyzsize; ++cs) {
      accum += (d_data[i * colsize + cs] - d_data[j * colsize + cs]) * (d_data[i * colsize + cs] - d_data[j * colsize + cs]);
    }
    
    if (label_mode) {
      if (accum < (eps[label]*eps[label]) && label == d_data[j * colsize + 4]) {
        d_Ea[basei] = j;
        ++basei;
      }
    }
    else
    {
      if (accum < (eps[0]*eps[0])) {
        d_Ea[basei] = j;
        ++basei;
      }
    }
  }
}

void asmadjlist(int N, int colsize, float* eps, float* d_data, int* d_Va1, int* d_Ea)
{
  _cu_asmadjlist<<<(N + 255) / 256, 256>>>(N, colsize, eps, d_data, d_Va1, d_Ea);
  //cudaDeviceSynchronize();
}

__global__ void
_cu_breadth_first_search_kern(int numpts, int* d_Ea, int* d_Va0, int* d_Va1, int* d_Fa, int* d_Xa)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= numpts)
    return;

  if (d_Fa[tid])
  {
    d_Fa[tid] = 0;
    d_Xa[tid] = 1;

    int nmax_idx = d_Va1[tid] + d_Va0[tid];

    for (int i = d_Va1[tid]; i < nmax_idx; ++i )
    {
      int nid = d_Ea[i];
      if (!d_Xa[nid])
      {
        d_Fa[nid] = 1;
      } 
    }
  }
}

void breadth_first_search_kern(int N, int* d_Ea, int* d_Va0, int* d_Va1, int* d_Fa, int* d_Xa)
{
  _cu_breadth_first_search_kern<<<(N + 255) / 256, 256>>>(N, d_Ea, d_Va0, d_Va1, d_Fa, d_Xa);
  //cudaDeviceSynchronize();
}

#pragma GCC diagnostic pop
