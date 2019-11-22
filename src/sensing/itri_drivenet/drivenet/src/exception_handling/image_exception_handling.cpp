#include "drivenet/image_exception_handling.h"

bool CheckMatDataValid(Mat src)
{
    if (!src.data)
    {
        std::cout << "Unable to read image." << std::endl;
        return false;
    }
    else if (src.cols <= 0 || src.rows <= 0)
    {
        std::cout << "Image cols: " << src.cols << ", rows: " << src.rows << std::endl;
        return false;
    }
    return true;
}
