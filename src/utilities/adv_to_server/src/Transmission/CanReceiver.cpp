#include "CanReceiver.h"
using namespace std;

#define NO_UNUSED_VAR_CHECK(x) ((void)(x))

CanReceiver::CanReceiver()
{
  c_socket = -1;
}

void CanReceiver::initial()
{
  int rc;
  struct can_filter filter[1];
  /*
    filter[0].can_id   = 0x350;
    filter[0].can_mask = CAN_SFF_MASK;
  */
  filter[0].can_id = 0x351;
  filter[0].can_mask = CAN_SFF_MASK;

  struct sockaddr_can addr;
  struct ifreq ifr;
  const char* ifname = "can1";
  if ((c_socket = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
  {
    perror("Error while opening socket");
    return;
  }
  rc = setsockopt(c_socket, SOL_CAN_RAW, CAN_RAW_FILTER, &filter, sizeof(filter));
  if (rc == -1)
  {
    std::perror("setsockopt filter");
    return;
  }
  strcpy(ifr.ifr_name, ifname);
  ioctl(c_socket, SIOCGIFINDEX, &ifr);
  addr.can_family = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;

  printf("%s at index %d\n", ifname, ifr.ifr_ifindex);

  if (bind(c_socket, (struct sockaddr*)&addr, sizeof(addr)) < 0)
  {
    perror("Error in socket bind");
  }
}

void CanReceiver::closeSocket()
{
  close(c_socket);
}

int CanReceiver::receive(double* data)
{
  int nbytes;
  struct can_frame frame;
  nbytes = read(c_socket, &frame, sizeof(frame));
  NO_UNUSED_VAR_CHECK(nbytes);
  int id = processFrame(data, frame);
  return id;
}

int CanReceiver::processFrame(double* data, const struct can_frame& frame)
{
  switch (frame.can_id)
  {
    /*
        case 0x350:
        {
          int lat_int;
          int lon_int;
          double mutiplier = pow(10.0, -7.0);
          lat_int = frame.data[0] | frame.data[1] << 8 | frame.data[2] << 16 | frame.data[3] << 24;
          lon_int = frame.data[4] | frame.data[5] << 8 | frame.data[6] << 16 | frame.data[7] << 24;
          data[0] = (double)lat_int * mutiplier;
          data[1] = (double)lon_int * mutiplier;
        }
        break;
    */
    case 0x351:
    {
      short speed;
      short f_brake_pressure;
      short r_brake_pressure;
      short steering_wheel_angle;
      short throttle_percentage;
      short break_percentage;
      double mutiplier = pow(10.0, -1.0);
      speed = frame.data[0];
      f_brake_pressure = frame.data[1];
      r_brake_pressure = frame.data[2];
      steering_wheel_angle = frame.data[3] | frame.data[4] << 8;
      throttle_percentage = frame.data[5];
      break_percentage = frame.data[6];
      data[0] = speed;
      data[1] = (double)f_brake_pressure * mutiplier;
      data[2] = (double)r_brake_pressure * mutiplier;
      data[3] = steering_wheel_angle * mutiplier;
      data[4] = throttle_percentage;
      data[5] = break_percentage;
    }
    break;
    default:
      break;
  }
  int id = frame.can_id;
  return id;
}
