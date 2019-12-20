#include <buffer_node.h>
#include <stdio.h>

class Buffer
{
private:
  unsigned int feature_num;
  int frame_num;
  int life;
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
