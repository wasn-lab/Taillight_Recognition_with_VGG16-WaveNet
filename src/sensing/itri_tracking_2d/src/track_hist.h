#ifndef __TRACK_HIST_H__
#define __TRACK_HIST_H__

#include "track2d.h"
#include <vector>
#include <msgs/BoxPoint.h>
#include <msgs/TrackState.h>

namespace track2d
{
class TrackHist
{
public:
  unsigned int id_ = 0;
  unsigned int tracktime_ = 0;
  unsigned char head_ = 255;
  unsigned short len_ = 0;

  static constexpr unsigned char max_len_ = 10;
  static constexpr int invalid_ = -10000;

  std_msgs::Header header_;

  std::vector<msgs::TrackState> states_;

  TrackHist()
  {
    init();
  }

  ~TrackHist()
  {
  }

  void reset();

  void set_for_first_element(const unsigned int id, const unsigned int tracktime, const float x, const float y,
                             const float estimated_x, const float estimated_y, const float estimated_velocity_x,
                             const float estimated_velocity_y);

  void set_for_successive_element(const unsigned int tracktime, const float x, const float y, const float estimated_x,
                                  const float estimated_y, const float estimated_velocity_x,
                                  const float estimated_velocity_y);

private:
  // DISALLOW_COPY_AND_ASSIGN(TrackHist);  // cause build error, root cause unknown

  void init();

  void set_state(const float x, const float y, const float estimated_x, const float estimated_y,
                 const float estimated_velocity_x, const float estimated_velocity_y);
};
}  // namespace track2d

#endif  // __TRACK_HIST_H__
