#ifndef __SENSING_GRABBER_PARSER_H__
#define __SENSING_GRABBER_PARSER_H__

#include <string>

namespace SensingSubSystem
{
// Getters
std::string get_mode();
int get_expected_fps();
bool should_feed_608();
};  // namespace

#endif  //__SENSING_GRABBER_PARSER_H__
