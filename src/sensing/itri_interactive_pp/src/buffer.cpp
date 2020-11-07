#include <buffer.h>

void Buffer::initial()
{
  first = 0;
  last = 0;
}

BufferNode* Buffer::is_in_the_list(int id)
{
  BufferNode* current_node = first;
  while (current_node != 0)
  {
    if (current_node->id == id)
    {
      return current_node;
    }
    current_node = current_node->next;
  }
  return 0;
}

std::vector<float> Buffer::add(int id, std::vector<float> feature)
{
  if (feature.size() != feature_num)
  {
    std::cout << "Wrong feature size!!!!!\n";
    std::vector<float> void_vec;
    return void_vec;
  }
  // Check if there is data in buffer.
  BufferNode* node_ptr = is_in_the_list(id);
  // There is data in buffer.
  // Delete T = t - 3 frame and add T = t frame
  if (node_ptr != 0)
  {
    node_ptr->data.erase(node_ptr->data.begin(), node_ptr->data.begin() + feature_num);
    node_ptr->data.insert(node_ptr->data.end(), feature.begin(), feature.end());
    node_ptr->refresh = true;
    return node_ptr->data;
  }
  // There is no data in buffer.
  // Add zero vector before T = t frame.
  // Then add T = t frame to buffer.
  else
  {
    BufferNode* node_ptr = new BufferNode(id);

    float* zero_arr;
    int other_feature = feature_num * (frame_num - 1);
    zero_arr = new float[other_feature]();
    feature.insert(feature.begin(), zero_arr, zero_arr + other_feature);
    delete[] zero_arr;
    node_ptr->data = feature;
    // Add link to other buffer_node
    if (first == 0 && last == 0)
    {
      first = node_ptr;
      last = node_ptr;
    }
    else
    {
      node_ptr->previous = last;
      last->next = node_ptr;
      last = node_ptr;
    }
    return node_ptr->data;
  }
}

void Buffer::remove(BufferNode* node_ptr)
{
  if (node_ptr != 0)
  {
    if (first == node_ptr)
    {
      first = node_ptr->next;
    }
    if (last == node_ptr)
    {
      last = node_ptr->previous;
    }
    if (node_ptr->previous != 0)
    {
      node_ptr->previous->next = node_ptr->next;
    }
    if (node_ptr->next != 0)
    {
      node_ptr->next->previous = node_ptr->previous;
    }
    delete node_ptr;
  }
}

void Buffer::check_life()
{
  BufferNode* current_node = first;
  while (current_node != 0)
  {
    BufferNode* next_node = current_node->next;
    if (current_node->refresh)
    {
      current_node->idle_time = 0;
      current_node->refresh = false;
    }
    else if (++current_node->idle_time > life)
    {
      remove(current_node);
    }
    current_node = next_node;
  }
}

void Buffer::display()
{
  BufferNode* current_node = first;
  std::cout << "Feature_num: " << feature_num << "\tFrame_num: " << frame_num << "\n\n";
  std::cout << "First: " << first << "\tLast: " << last << "\n\n";
  while (current_node != 0)
  {
    std::cout << "ID: " << current_node->id << "\tIdle Time: " << current_node->idle_time << "\tAll_feature_num"
              << current_node->data.size() << "\n";
    // for (int i = 0; i < current_node->data.size(); i++)
    // std::cout << current_node->data[i] << "\t";
    std::cout << std::endl;
    current_node = current_node->next;
  }
  std::cout << "------------------------------------------------------\n";
}
