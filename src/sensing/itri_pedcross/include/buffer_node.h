#include "ros/ros.h"
#include <vector>

class BufferNode
{
public:
  int id;
  std::vector<float> data;
  bool refresh;
  int idle_time;
  BufferNode* previous;
  BufferNode* next;
  BufferNode(int x);
};
