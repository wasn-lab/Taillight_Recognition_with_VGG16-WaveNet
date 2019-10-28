#ifndef CUDA_CTX
#define CUDA_CTX value

#include <cuda.h>
#include <memory>
#include <string.h>

class CudaCtx
{
  private:
    static const int def_device = 0;

  public:
    CudaCtx () :
        dev_handle (new int (0)),
        dev_cnt (new int (0)),
        mem_size (new size_t (0))
    {
      CUresult r;
      r = cuInit (0);

      if (r != CUDA_SUCCESS)
      {
        throw std::runtime_error ("Cuda init error :" + std::to_string (r));
      }

      r = cuDeviceGet (dev_handle.get (), 0);

      if (r != CUDA_SUCCESS)
      {
        throw std::runtime_error ("Cuda device get error :" + std::to_string (r));
      }

      r = cuDeviceGetCount (dev_cnt.get ());

      if (r != CUDA_SUCCESS)
      {
        throw std::runtime_error ("Cuda device cnt error :" + std::to_string (r));
      }

      r = cuDeviceTotalMem (mem_size.get (), *dev_handle);

      if (r != CUDA_SUCCESS)
      {
        throw std::runtime_error ("Cuda device total mem error :" + std::to_string (r));
      }

      char dev_name[256];
      r = cuDeviceGetName (dev_name, sizeof (dev_name), *dev_handle);

      if (r != CUDA_SUCCESS)
      {
        throw std::runtime_error ("Cuda device name error :" + std::to_string (r));
      }

      dev_name[255] = 0;

      device_name.assign (dev_name, strlen (dev_name));
    }

    ~CudaCtx ()
    {

    }

    const std::string
    getDevName () const
    {
      return device_name;
    }

    int
    getDevCount () const
    {
      return *dev_cnt;
    }

    size_t
    getDevMemory () const
    {
      return *mem_size;
    }



  private:
    std::unique_ptr<CUdevice> dev_handle;
    std::unique_ptr<int> dev_cnt;
    std::unique_ptr<size_t> mem_size;
    std::string device_name;
};

#endif
