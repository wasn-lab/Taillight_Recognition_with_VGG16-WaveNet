#include <buffer_node.h>
#include <cstdio>

class Buffer
{
private:
  // feature vector: 1~4 BBox, 5~316 distance of each two points,
  // 317~1174 inner angle of each three points
  const unsigned int feature_num = 1174;
  const int frame_num = 10;
  const int life = 5;
  BufferNode* first;
  BufferNode* last;

public:
  ros::Time timestamp;

  void initial();

  BufferNode* is_in_the_list(int id);

  std::vector<float> add(int id, std::vector<float> feature);

  void remove(BufferNode* node_ptr);

  void check_life();

  void display();
};
