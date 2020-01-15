#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>

#include <linux/can.h>
#include <linux/can/raw.h>


int main(void) 
{
	int s;
	int nbytes;
	struct sockaddr_can addr;
	struct can_frame frame;
	struct ifreq ifr;

	const char *ifname = "can0";

	if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) {
		perror("Error while opening socket");
		return -1;
	}

	strcpy(ifr.ifr_name, ifname);
	ioctl(s, SIOCGIFINDEX, &ifr);
	
	addr.can_family  = AF_CAN;
	addr.can_ifindex = ifr.ifr_ifindex;

	printf("%s at index %d\n", ifname, ifr.ifr_ifindex);

	if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		perror("Error in socket bind");
		return -2;   
	}

	frame.can_id  = 0x30;	//
	frame.can_dlc = 8;	//receive byte
	
	/*
	  struct can_frame {
	  	canid_t can_id;  /* 32 bit CAN_ID + EFF/RTR/ERR flags
	  	__u8    can_dlc; /* frame payload length in byte (0 .. CAN_MAX_DLEN)
	  	__u8    __pad;   /* padding
	  	__u8    __res0;  /* reserved / padding
	  	__u8    __res1;  /* reserved / padding
	  	__u8    data[CAN_MAX_DLEN] __attribute__((aligned(8)));
	  };
	*/

	while(1)
		{
			nbytes = read(s, &frame, sizeof(struct can_frame));
			printf("Read %d bytes\n", nbytes);
			printf("id: %03x, dlc: %d\n", frame.can_id, frame.can_dlc);
			for(int i=0; i<frame.can_dlc;i++)
			    printf("%02x ", frame.data[i]);
			printf("\n");
			// printf("%d, %f, %f\n", frame.data[0], (float)frame.data[1]+frame.data[2]/100.0, (float)frame.data[3]+frame.data[4]/100.0);
		}
	
	return 0;
}

