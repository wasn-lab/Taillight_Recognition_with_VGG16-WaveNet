#include <assert.h>
#include <fstream>
#include <iostream>
#include <jsoncpp/json/json.h>
#include <alignment_json_reader.h>

namespace alignment
{
int read_distance_from_json(const std::string& filename, cv::Point3d** dist_in_cm, const int rows, const int cols)
{
  // dist_in_cm should be malloc by caller.
  assert(dist_in_cm);
  for (int i = 0; i < rows; i++)
  {
    assert(dist_in_cm[i]);
  }

  std::ifstream ifs(filename);
  Json::Reader jreader;
  Json::Value jdata;
  jreader.parse(ifs, jdata);

  for (Json::ArrayIndex i = 0; i < jdata.size(); i++)
  {
    auto image_x = jdata[i]["im_x"].asInt();
    auto image_y = jdata[i]["im_y"].asInt();
    assert(image_x >= 0);
    assert(image_y >= 0);

    if ((image_y < rows) && (image_x < cols))
    {
      dist_in_cm[image_y][image_x].x = jdata[i]["dist_in_cm"][0].asInt();
      dist_in_cm[image_y][image_x].y = jdata[i]["dist_in_cm"][1].asInt();
      dist_in_cm[image_y][image_x].z = jdata[i]["dist_in_cm"][2].asInt();
    }
  }
  return 0;
}

};  // namespace
