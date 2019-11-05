#ifndef WEATHER_HPP
#define WEATHER_HPP

#include "opencv2/highgui/highgui.hpp"  //for reading and visualizing image
#include "opencv2/imgproc/imgproc.hpp"  //for image resizing

//Defining PI value
#ifndef PI
#ifdef M_PI
#define PI M_PI
#else
#define PI 3.14159265358979
#endif
#endif

extern cv::Mat cam10_cloneImg;
extern cv::Mat cam11_cloneImg;
extern cv::Mat cam12_cloneImg;


typedef struct _scanlimit scanlimit_t;
struct _scanlimit{
	int x;
	int y;
	double nRoad;
	double nSky;
};
typedef struct _lux lux_t;
struct _lux{
	int** sky;
	int** road;
	bool high;
};
typedef struct _wea wea_t;
struct _wea{
	double* sky;
	double* road;
	bool norm;
};

typedef struct _image3 image3_t;
struct _image3{
	int*** pixel;
	int width, height;
};

typedef struct _image image_t;
struct _image{
	int* pixel;
	int width, height;
};

void* pthrd_intensity_measure(void* argv);

void init_intensity_measure();

void cvMat2LuxSat(cv::Mat input, image_t &lummap, image_t &satmap);
//luminance
void luxmeter(image_t lummap, scanlimit_t scan, lux_t &lums);
//exposure
void expmeter(image_t lummap, image_t satmap, image_t &overmap, image_t &undrmap);
//weather
void weathermeter(image_t lummap, scanlimit_t scan, lux_t &lums, wea_t &stds);

#endif // WEATHER_HPP

