#include "track_hist.h"

namespace tpp
{
void TrackHist::init()
{
  reset();

  states_.reserve(max_len_);

  msgs::TrackState state;
  state.position.x = invalid_;
  state.position.y = invalid_;
  state.position.z = invalid_;

  state.estimated_position.x = invalid_;
  state.estimated_position.y = invalid_;
  state.estimated_position.z = invalid_;

  state.estimated_velocity.x = invalid_;
  state.estimated_velocity.y = invalid_;
  state.estimated_velocity.z = invalid_;

  for (unsigned i = 0; i < max_len_; i++)
  {
    states_.push_back(state);
  }
}

void TrackHist::reset()
{
  id_ = 0;
  tracktime_ = 0;
  head_ = 255;
  len_ = 0;
}

void TrackHist::set_state(const float x, const float y, const float estimated_x, const float estimated_y,
                          const float estimated_velocity_x, const float estimated_velocity_y)
{
  states_[head_].header = header_;
  states_[head_].position.x = x;
  states_[head_].position.y = y;
  states_[head_].estimated_position.x = estimated_x;
  states_[head_].estimated_position.y = estimated_y;
  states_[head_].estimated_velocity.x = estimated_velocity_x;
  states_[head_].estimated_velocity.y = estimated_velocity_y;
}

void TrackHist::set_for_first_element(const unsigned int id, const unsigned int tracktime, const float x, const float y,
                                      const float estimated_x, const float estimated_y,
                                      const float estimated_velocity_x, const float estimated_velocity_y)
{
  id_ = id;
  tracktime_ = tracktime;
  head_ = 0;
  len_ = 1;

  set_state(x, y, estimated_x, estimated_y, estimated_velocity_x, estimated_velocity_y);
}

void TrackHist::set_for_successive_element(const unsigned int tracktime, const float x, const float y,
                                           const float estimated_x, const float estimated_y,
                                           const float estimated_velocity_x, const float estimated_velocity_y)
{
  tracktime_ = tracktime;
  head_ = (head_ + 1) % max_len_;

  if (len_ < (unsigned short)max_len_)
    len_++;

  set_state(x, y, estimated_x, estimated_y, estimated_velocity_x, estimated_velocity_y);
}
}  // namespace tpp