#ifndef CANMODULE_H
#define CANMODULE_H

#include "../all_header.h"

/* example
 *
 * CanModule::initial ();
 *
 */


class CanModule
{
  public:

    static int device;
    static size_t required_mtu;
    static unsigned char counter;

    CanModule ()
    {
      counter = 0;
    }

    ~CanModule ()
    {
      close (device);
    }

    static void
    initial ()
    {
      int mtu;
      int enable_canfd = 1;
      struct sockaddr_can addr;
      struct ifreq ifr;

      if ( (device = socket (PF_CAN, SOCK_RAW, CAN_RAW)) < 0)
      {
        perror ("[CAN] Error while opening socket");
      }

      required_mtu = CAN_MTU;
      //required_mtu = CANFD_MTU;

      strncpy (ifr.ifr_name, "can0", IFNAMSIZ - 1);
      ifr.ifr_name[IFNAMSIZ - 1] = '\0';
      ifr.ifr_ifindex = if_nametoindex (ifr.ifr_name);
      if (!ifr.ifr_ifindex)
      {
        perror ("[CAN] if_nametoindex");
      }

      addr.can_family = AF_CAN;
      addr.can_ifindex = ifr.ifr_ifindex;

      if (required_mtu > CAN_MTU)
      {
        //check if the frame fits into the CAN netdevice
        if (ioctl (device, SIOCGIFMTU, &ifr) < 0)
        {
          perror ("[CAN] SIOCGIFMTU");
          //return 1;
        }
        mtu = ifr.ifr_mtu;

        if (mtu != CANFD_MTU)
        {
          printf ("[CAN] interface is not CAN FD capable - sorry.\n");
        }

        //interface is ok - try to switch the socket into CAN FD mode
        if (setsockopt (device, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &enable_canfd, sizeof (enable_canfd)))
        {
          printf ("[CAN] error when enabling CAN FD support\n");
        }

        //ensure discrete CAN FD length values 0..8, 12, 16, 20, 24, 32, 64
        //frame.len = can_dlc2len(can_len2dlc(frame.len));
      }

      /*
       disable default receive filter on this RAW socket
       This is obsolete as we do not read from the socket at all, but for
       this reason we can remove the receive list in the Kernel to save a
       little (really a very little!) CPU usage.
       */
      setsockopt (device, SOL_CAN_RAW, CAN_RAW_FILTER, NULL, 0);

      if (bind (device, (struct sockaddr *) &addr, sizeof (addr)) < 0)
      {
        perror ("[CAN] bind");
      }

    }

    static void
    ReadAndWrite_controller (CLUSTER_INFO* cluster_info,
                             int cluster_size)
    {
      /*
       struct can_frame {
       canid_t can_id;  // 32 bit CAN_ID + EFF/RTR/ERR flags
       __u8    can_dlc; // frame payload length in byte (0 .. CAN_MAX_DLEN)
       __u8    __pad;   // padding
       __u8    __res0;  // reserved / padding
       __u8    __res1;  // reserved / padding
       __u8    data[CAN_MAX_DLEN] __attribute__((aligned(8)));
       };
       */

      if (device > 0)
      {

        CLUSTER_INFO buff[10] = { };
        int stacked_num = 0;

        if (cluster_size > 0 && cluster_size <= 10)
        {
#pragma omp parallel for
          for (int i = 0; i < cluster_size; i++)
          {
            if (cluster_info[i].center.y > 0 && cluster_info[i].center.y < 12 && fabs (cluster_info[i].center.x) < 5)
            {
              buff[i] = cluster_info[i];
              stacked_num++;
            }
          }
        }
        else if (cluster_size > 10)
        {
          // limit the range of cluster_info, and get modify_cluster_info
          vector<CLUSTER_INFO> modify_cluster_info;

          for (int i = 0; i < cluster_size; i++)
          {
            if (cluster_info[i].center.y > 0 && cluster_info[i].center.y < 12 && fabs (cluster_info[i].center.x) < 5)
            {
              modify_cluster_info.push_back (cluster_info[i]);
            }
          }

          // Start scan area

          for (size_t i = 0; i < modify_cluster_info.size (); i++)
          {
            if (stacked_num == 10)
              break;

            if (modify_cluster_info.at (i).center.y <= 3)
            {
              buff[stacked_num] = modify_cluster_info.at (i);
              modify_cluster_info.erase (modify_cluster_info.begin () + i);
              stacked_num++;
            }
          }

          for (size_t i = 0; i < modify_cluster_info.size (); i++)
          {
            if (stacked_num == 10)
              break;

            if (modify_cluster_info.at (i).center.y <= 6)
            {
              buff[stacked_num] = modify_cluster_info.at (i);
              modify_cluster_info.erase (modify_cluster_info.begin () + i);
              stacked_num++;
            }
          }

          for (size_t i = 0; i < modify_cluster_info.size (); i++)
          {
            if (stacked_num == 10)
              break;

            if (modify_cluster_info.at (i).center.y <= 9)
            {
              buff[stacked_num] = modify_cluster_info.at (i);
              modify_cluster_info.erase (modify_cluster_info.begin () + i);
              stacked_num++;
            }
          }

          for (size_t i = 0; i < modify_cluster_info.size (); i++)
          {
            if (stacked_num == 10)
              break;

            if (modify_cluster_info.at (i).center.y <= 12)
            {
              buff[stacked_num] = modify_cluster_info.at (i);
              modify_cluster_info.erase (modify_cluster_info.begin () + i);
              stacked_num++;
            }
          }
        }

        int send_result = 0;

        struct canfd_frame frame;
        frame.can_id = 0x400;
        frame.len = 8;
        frame.data[0] = counter;
        frame.data[1] = stacked_num;
        frame.data[2] = 0;
        frame.data[3] = 0;
        frame.data[4] = 0;
        frame.data[5] = 0;
        frame.data[6] = 0;
        frame.data[7] = 0;
        send_result += write (device, &frame, required_mtu);
        counter++;

        for (int i = 0; i < 5; i++)
        {
          frame.can_id++;
          frame.data[0] = (int) (buff[i * 2].center.x * 10);
          frame.data[1] = (int) (buff[i * 2].center.y * 10);
          frame.data[2] = (int) (buff[i * 2].dx * 10);
          frame.data[3] = (int) (buff[i * 2].dy * 10);
          frame.data[4] = (int) (buff[i * 2 + 1].center.x * 10);
          frame.data[5] = (int) (buff[i * 2 + 1].center.y * 10);
          frame.data[6] = (int) (buff[i * 2 + 1].dx * 10);
          frame.data[7] = (int) (buff[i * 2 + 1].dy * 10);

          send_result += write (device, &frame, required_mtu);
        }
        cout << "[CAN]: write " << send_result << endl;
        cout << "[CAN]: num " << stacked_num << endl;

        // You can you select or other standard API for waiting CAN frame  to read for example..

        //struct can_frame frame_rd;
        //ssize_t nbytesRD = read (device, &frame_rd, sizeof(struct can_frame));  // Read a CAN frame
        //cout << "[SocketCAN]: read " << nbytesRD;

      }
    }

    static void
    ReadAndWrite_controller_v2 (CLUSTER_INFO* cluster_info,
                                int cluster_size)
    {
      /*
       struct can_frame {
       canid_t can_id;  // 32 bit CAN_ID + EFF/RTR/ERR flags
       __u8    can_dlc; // frame payload length in byte (0 .. CAN_MAX_DLEN)
       __u8    __pad;   // padding
       __u8    __res0;  // reserved / padding
       __u8    __res1;  // reserved / padding
       __u8    data[CAN_MAX_DLEN] __attribute__((aligned(8)));
       };
       */

      if (device > 0)
      {

        CLUSTER_INFO buff[10] = { };
        int stacked_num = 0;

        if (cluster_size > 0 && cluster_size <= 10)
        {
#pragma omp parallel for
          for (int i = 0; i < cluster_size; i++)
          {
            if (cluster_info[i].center.y > 0 && cluster_info[i].center.y < 12 && fabs (cluster_info[i].center.x) < 5)
            {
              buff[i] = cluster_info[i];
              stacked_num++;
            }
          }
        }
        else if (cluster_size > 10)
        {
          // limit the range of cluster_info, and get modify_cluster_info
          vector<CLUSTER_INFO> modify_cluster_info;

          for (int i = 0; i < cluster_size; i++)
          {
            if (cluster_info[i].center.y > 0 && cluster_info[i].center.y < 12 && fabs (cluster_info[i].center.x) < 5)
            {
              modify_cluster_info.push_back (cluster_info[i]);
            }
          }

          // Start scan area

          for (size_t i = 0; i < modify_cluster_info.size (); i++)
          {
            if (stacked_num == 10)
              break;

            if (modify_cluster_info.at (i).center.y <= 3)
            {
              buff[stacked_num] = modify_cluster_info.at (i);
              modify_cluster_info.erase (modify_cluster_info.begin () + i);
              stacked_num++;
            }
          }

          for (size_t i = 0; i < modify_cluster_info.size (); i++)
          {
            if (stacked_num == 10)
              break;

            if (modify_cluster_info.at (i).center.y <= 6)
            {
              buff[stacked_num] = modify_cluster_info.at (i);
              modify_cluster_info.erase (modify_cluster_info.begin () + i);
              stacked_num++;
            }
          }

          for (size_t i = 0; i < modify_cluster_info.size (); i++)
          {
            if (stacked_num == 10)
              break;

            if (modify_cluster_info.at (i).center.y <= 9)
            {
              buff[stacked_num] = modify_cluster_info.at (i);
              modify_cluster_info.erase (modify_cluster_info.begin () + i);
              stacked_num++;
            }
          }

          for (size_t i = 0; i < modify_cluster_info.size (); i++)
          {
            if (stacked_num == 10)
              break;

            if (modify_cluster_info.at (i).center.y <= 12)
            {
              buff[stacked_num] = modify_cluster_info.at (i);
              modify_cluster_info.erase (modify_cluster_info.begin () + i);
              stacked_num++;
            }
          }
        }

        int send_result = 0;

        struct canfd_frame frame;
        frame.can_id = 0x400;
        frame.len = 8;
        frame.data[0] = counter;
        frame.data[1] = stacked_num;
        frame.data[2] = 0;
        frame.data[3] = 0;
        frame.data[4] = 0;
        frame.data[5] = 0;
        frame.data[6] = 0;
        frame.data[7] = 0;
        send_result += write (device, &frame, required_mtu);
        counter++;

        for (int i = 0; i < 10; i++)
        {
          frame.can_id++;
          frame.data[0] = (int) (buff[i * 2].obb_center.x * 10);
          frame.data[1] = (int) (buff[i * 2].obb_center.y * 10);
          frame.data[2] = (int) (buff[i * 2].obb_dx * 10);
          frame.data[3] = (int) (buff[i * 2].obb_dy * 10);
          frame.data[4] = (int) (buff[i * 2].obb_orient);
          frame.data[5] = (int) (buff[i * 2].velocity.x * 10);
          frame.data[6] = (int) (buff[i * 2].velocity.y * 10);
          frame.data[7] = (int) (buff[i * 2].tracking_id);

          send_result += write (device, &frame, required_mtu);
        }
        cout << "[CAN]: write " << send_result << endl;
        cout << "[CAN]: num " << stacked_num << endl;

        // You can you select or other standard API for waiting CAN frame  to read for example..

        //struct can_frame frame_rd;
        //ssize_t nbytesRD = read (device, &frame_rd, sizeof(struct can_frame));  // Read a CAN frame
        //cout << "[SocketCAN]: read " << nbytesRD;

      }
    }

};

int CanModule::device;
size_t CanModule::required_mtu;
unsigned char CanModule::counter;

#endif // CANMODULE_H
