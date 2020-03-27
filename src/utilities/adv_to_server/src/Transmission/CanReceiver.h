#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <cmath>

#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>

#include <linux/can.h>
#include <linux/can/raw.h>

#define CAN_DLC 8;

class CanReceiver

{
public:
  CanReceiver();

  void initial();

  void closeSocket();

  int receive(double* data);

  int processFrame(double* data, const struct can_frame& frame);

private:
  int c_socket;
};
