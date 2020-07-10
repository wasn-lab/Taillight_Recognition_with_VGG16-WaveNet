/*
 *   File: localization_can_class.h
 *   Created on: April, 2018
 *   Author: Xu, Bo Chun & Wayne Yang
 *   Institute: ITRI ICL U300
 */

#include "localization_can_class.h"

#define CAN_INTERFACE_NAME "can0"

ClassLiDARPoseCan::ClassLiDARPoseCan()
{
}

ClassLiDARPoseCan::~ClassLiDARPoseCan()
{
  close(s);
}

void ClassLiDARPoseCan::initial()
{
  int mtu;
  int enable_canfd = 1;
  struct sockaddr_can addr;
  struct ifreq ifr;

  if ((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
  {
    perror("[CAN] Error while opening socket");
  }

  required_mtu = CAN_MTU;
  // required_mtu = CANFD_MTU;

  strncpy(ifr.ifr_name, CAN_INTERFACE_NAME, IFNAMSIZ - 1);
  ifr.ifr_name[IFNAMSIZ - 1] = '\0';
  ifr.ifr_ifindex = if_nametoindex(ifr.ifr_name);
  if (!ifr.ifr_ifindex)
  {
    perror("[CAN] if_nametoindex");
  }

  addr.can_family = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;

  if (required_mtu > CAN_MTU)
  {
    // check if the frame fits into the CAN netdevice
    if (ioctl(s, SIOCGIFMTU, &ifr) < 0)
    {
      perror("[CAN] SIOCGIFMTU");
      // return 1;
    }
    mtu = ifr.ifr_mtu;

    if (mtu != CANFD_MTU)
    {
      printf("[CAN] interface is not CAN FD capable - sorry.\n");
    }

    // interface is ok - try to switch the socket into CAN FD mode
    if (setsockopt(s, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &enable_canfd, sizeof(enable_canfd)))
    {
      printf("[CAN] error when enabling CAN FD support\n");
    }

    // ensure discrete CAN FD length values 0..8, 12, 16, 20, 24, 32, 64
    // frame.len = can_dlc2len(can_len2dlc(frame.len));
  }
  /*
     disable default receive filter on this RAW socket
     This is obsolete as we do not read from the socket at all, but for
     this reason we can remove the receive list in the Kernel to save a
     little (really a very little!) CPU usage.
   */
  setsockopt(s, SOL_CAN_RAW, CAN_RAW_FILTER, NULL, 0);

  if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) < 0)
  {
    perror("[CAN] bind");
  }
}

//    /*
//      struct can_frame {
//        canid_t can_id;  /* 32 bit CAN_ID + EFF/RTR/ERR flags
//        __u8    can_dlc; /* frame payload length in byte (0 ..
//      CAN_MAX_DLEN)
//        __u8    __pad;   /* padding
//        __u8    __res0;  /* reserved / padding
//        __u8    __res1;  /* reserved / padding
//        __u8    data[CAN_MAX_DLEN] __attribute__((aligned(8)));
//      };
//      */

int ClassLiDARPoseCan::poseSendByCAN(const struct MsgSendToCan& input_msg)
{
  struct canfd_frame frame;
  frame.can_id = 0x460;
  frame.len = 8;
  int nbytes = 0;

  if (can_counter > 255)
  {
    can_counter = 0;
  }

  frame.data[0] = (int)(input_msg.x * 100) & 0xff;
  frame.data[1] = (int)(input_msg.x * 100) >> 8;
  frame.data[2] = (int)(input_msg.x * 100) >> 16;

  frame.data[3] = (int)(input_msg.y * 100) & 0xff;
  frame.data[4] = (int)(input_msg.y * 100) >> 8;
  frame.data[5] = (int)(input_msg.y * 100) >> 16;

  frame.data[6] = lidar_error;
  frame.data[7] = can_counter;

  nbytes += write(s, &frame, required_mtu);

  frame.can_id++;
  // HEADING RAD
  frame.data[0] = (int)(input_msg.heading * 100) & 0xff;
  frame.data[1] = (int)(input_msg.heading * 100) >> 8;
  frame.data[2] = (int)(input_msg.fitness_score * 100) & 0xff;
  frame.data[3] = (int)(input_msg.fitness_score * 100) >> 8;
  frame.data[4] = (int)(input_msg.transform_prob * 100) & 0xff;
  frame.data[5] = (int)(input_msg.transform_prob * 100) >> 8;

  frame.data[6] = can_counter;
  frame.data[7] = 0;
  nbytes += write(s, &frame, required_mtu);

  cout << "[CAN]: wrote " << nbytes << endl;
  can_counter++;
}
