#include <assert.h>
#include <jsoncpp/json/json.h>
#include <alignment_json_writer.h>

namespace alignment
{
std::string jsonize_spatial_points(cv::Point3d** spatial_points, const int rows, const int cols)
{
  assert(spatial_points);
  assert(spatial_points[0]);
  Json::Value jspatial_points(Json::arrayValue);
  Json::FastWriter jwriter;
  for (int row = 0; row < rows; row++)
  {
    for (int col = 0; col < cols; col++)
    {
      Json::Value jvalue;
      Json::Value dist_in_cm(Json::arrayValue);
      jvalue["im_x"] = col;
      jvalue["im_y"] = row;

      // distance is measured in meters
      dist_in_cm[0] = int(spatial_points[row][col].x * 100);
      dist_in_cm[1] = int(spatial_points[row][col].y * 100);
      dist_in_cm[2] = int(spatial_points[row][col].z * 100);
      jvalue["dist_in_cm"] = dist_in_cm;

      jspatial_points.append(jvalue);
    }
  }

  jwriter.omitEndingLineFeed();
  return jwriter.write(jspatial_points);
}
};  // namespace
