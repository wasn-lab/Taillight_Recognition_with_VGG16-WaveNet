#ifndef OBJECTLABELUTIL_H_
#define OBJECTLABELUTIL_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int translate_label(int label);
cv::Scalar get_labelColor(std::vector<cv::Scalar> colors, int label_id);

#endif /*OBJECTLABELUTIL_H_*/