#include "drivenet/object_label_util.h"

int translate_label(int label) {
    if(label == 0){
        return 1;
    }else if(label == 1){
        return 2;        
    }else if(label == 2){
        return 4;
    }else if(label == 3){
        return 3;
    }else if(label == 5){
        return 5;
    }else if(label == 7){
        return 6;
    }else {
        return 0;
    }
}
cv::Scalar get_labelColor(std::vector<cv::Scalar> colors, int label_id)
{
    cv::Scalar class_color;
    if(label_id == 0)
        class_color = colors[0];
    else if (label_id == 1 || label_id == 3)
        class_color = colors[1];
    else if (label_id == 2 || label_id == 5 || label_id == 7)
        class_color = colors[2];
    else
        class_color = colors[3];
    return class_color;
}