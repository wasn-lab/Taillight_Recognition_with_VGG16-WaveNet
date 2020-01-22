#include <buffer_node.h>

BufferNode::BufferNode(int x)
{
  id = x;
  refresh = true;
  idle_time = 0;
  previous = 0;
  next = 0;
}
