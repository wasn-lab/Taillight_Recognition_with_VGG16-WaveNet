#ifndef CUDA_CTX
#define CUDA_CTX value

#include <cuda.h>
#include <memory>
#include <string.h>

class CudaCtx
{
  public:
    CudaCtx () :
        dev_handle (new int (0)),
        dev_cnt (new int (0)),
        mem_size (new size_t (0))
    {

      ErrorHandle(cuInit (0), "init");

      ErrorHandle(cuDeviceGet (dev_handle.get (), 0), "device");

      ErrorHandle(cuDeviceGetCount (dev_cnt.get ()), "device cnt");

      ErrorHandle(cuDeviceTotalMem (mem_size.get (), *dev_handle), "device total mem");

      char dev_name[256];

      ErrorHandle(cuDeviceGetName (dev_name, sizeof (dev_name), *dev_handle), "device name");

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
    static const int def_device = 0;

    std::unique_ptr<CUdevice> dev_handle;
    std::unique_ptr<int> dev_cnt;
    std::unique_ptr<size_t> mem_size;
    std::string device_name;

    void
    ErrorHandle(CUresult r, std::string Msg){
      if (r != CUDA_SUCCESS)
      {
        throw std::runtime_error ("[CUDA Driver] Error :" + Msg + ", " + std::to_string (r));
      }
    }
};

#endif
