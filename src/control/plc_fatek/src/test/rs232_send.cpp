/* rs232_send.c */
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define BAUDRATE B9600
#define MODEMDEVICE "/dev/ttyS1"
#define _POSIX_SOURCE 1
#define STOP '@'
#define bzero(b,len) (memset((b), '\0', (len)), (void) 0)

int main()
{
    int fd, c=0, res;
    struct termios oldtio, newtio;
    char ch;
    static char s1[20];

    printf("Start...\n");
    fd = open(MODEMDEVICE, O_RDWR|O_NOCTTY);
    if (fd < 0) {
      perror(MODEMDEVICE);
      exit(1);
    }

    printf("Open...\n");
    tcgetattr(fd, &oldtio);
    bzero(&newtio, sizeof(newtio));

    newtio.c_cflag = BAUDRATE|CS8|CLOCAL|CREAD;
    newtio.c_iflag = IGNPAR;
    newtio.c_oflag = 0;
    newtio.c_lflag = ICANON;

    tcflush(fd, TCIFLUSH);
    tcsetattr(fd, TCSANOW, &newtio);

    printf("Writing...\n");

    
    char str1[]={'\x02','\x30','\x33','\x34','\x30','\x43','\x37','\x03'};
    char str2[]="\x0201411F9\x03";
    while(1) {
        //while((ch=getchar())!= STOP){
            //s1[0]=ch;
            res=write(fd,str1,sizeof(str1));
            printf("write\n");
            usleep(3000000);
        }
        /*
        s1[0]=ch;
        s1[1]='\n';
        res = write(fd,s1,2);
        printf("\nWrite over");
        break;
        

        
    }*/

    printf("Close...\n");
    close(fd);
    tcsetattr(fd, TCSANOW, &oldtio);
    return 0;
}