#include "drivenet/image_exception_handling.h"

bool CheckMatDataValid(Mat src)
{
    if (src.empty())
    {
        std::cout << "Unable to read image." << std::endl;
        return false;
    }
    return true;
}
