### Install parknet dependencies

* Install gflags, glog and boost:
```
apt-get install libgflags-dev libgoogle-glog-dev libboost-all-dev
```

* Install Cuda 10.0

Follow the instruction [here](https://gitlab.itriadv.co/self_driving_bus/automatic_scripts/blob/master/cuda-10.0.sh)

* Install TensorRT:

TensorRT installation file is 831MB. It can be downloaded by the URL below.

```
wget "http://118.163.54.109:8888/Share/ADV/3rd source code/TensorRT/nv-tensorrt-repo-ubuntu1604-cuda10.0-trt5.0.2.6-ga-20181009_1-1_amd64.deb"
wget "http://118.163.54.109:8888/Share/ADV/3rd source code/TensorRT/TensorRT-Installation-Guide.pdf"
```
Follow TensorRT-Installation-Guide.pdf to install TensorRT.
