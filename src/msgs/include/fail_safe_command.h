#pragma once

namespace fail_safe_command
{
enum TargetId
{
  no_op = 0,  // no effect in the operation of car, this field is used for testing.
  speedup = 1,
  lane_change = 2,
  self_driving_mode = 3,  // turn on/off self-driving mode. No effect at this moment.
};

enum ActionId
{
  disable = 0,
  enable = 1,
  disallow = 0,
  allow = 1,
  off = 0,
  on = 1,
  turn_off = 0,
  turn_on = 1,
};
};  // namespace fail_safe_command
