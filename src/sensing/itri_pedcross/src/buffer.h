#include <buffer_node.h>
#include <stdio.h>

class Buffer
{
private:
  unsigned int feature_num = 1174;
  int frame_num = 3;
  int life = 5;
  BufferNode* first;
  BufferNode* last;

public:
  ros::Time timestamp;

  void initial(int x, int y, int z);

  BufferNode* is_in_the_list(int id);

  std::vector<float> add(int id, std::vector<float> feature);

  void remove(BufferNode* node_ptr);

  void check_life();

  void display();
};
