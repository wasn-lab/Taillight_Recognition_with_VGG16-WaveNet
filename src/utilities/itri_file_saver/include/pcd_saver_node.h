#pragma once

#include <memory>

class PCDSaverNodeImpl;

class PCDSaverNode
{
private:
  std::unique_ptr<PCDSaverNodeImpl> pcd_saver_node_impl_;

public:
  PCDSaverNode();
  ~PCDSaverNode();
  void run();
};

