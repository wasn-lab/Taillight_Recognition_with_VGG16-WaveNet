#ifndef RECTCLASSSCORE_H_
#define RECTCLASSSCORE_H_

#include <sstream>
#include <string>

namespace Yolo3
{
enum YoloDetectorClasses  // using coco for default cfg and weights
{
  corner,
};
}

template <typename _Tp>
class RectClassScore
{
public:
  _Tp x, y, w, h;
  _Tp score;
  unsigned int class_type;
  bool enabled;

  inline std::string toString() const
  {
    std::ostringstream out;
    out << class_type << "(x:" << x << ", y:" << y << ", w:" << w << ", h:" << h << ") =" << score;
    return out.str();
  }
  inline std::string GetClassString() const
  {
    switch (class_type)
    {
      case Yolo3::corner:
        return "Corner";
      default:
        return "Corner";
    }
  }
  inline int GetClassInt()
  {
    switch (class_type)
    {
      case Yolo3::corner:
        return 0;
      default:
        return 0;
    }
  }
};

#endif /* RECTCLASSSCORE_H_ */
