PCD Viewer
====

A simple PCD (Point Cloud Data) file viewer using glfw.

![image](https://raw.githubusercontent.com/7ew1r/pcd-viewer/images/image.png)

## Requirement

- PCL (Point Cloud Library) >= 1.7.2
- glfw

## Install

1. Install requirements

```
$ sudo apt install libpcl-dev glfw libglfw3-dev cmake
```

2. Clone repository & build

```
$ cd /dir/to/repo
$ mkdir build && cd build
$ cmake .. && make
```

## Usage

```
$ ./PCD_Viewer
```

Drag and drop the PCD file  on window to import object.  
You can rotate an object by drag on the window.
