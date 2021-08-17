#include     <stdio.h>      /*標準輸入輸出定義*/

#include     <stdlib.h>     /*標準函數庫定義*/

#include     <unistd.h>     /*Unix標準函數定義*/

#include     <sys/types.h>  /**/

#include     <sys/stat.h>   /**/

#include     <fcntl.h>      /*檔控制定義*/

#include     <termios.h>    /*PPSIX終端控制定義*/

#include     <errno.h>      /*錯誤號定義*/

 

/***@brief  設置串口通信速率

*@param  fd     類型 int  打開串口的文件控制碼

*@param  speed  類型 int  串口速度

*@return  void*/

 

int speed_arr[] = { B38400, B19200, B9600, B4800, B2400, B1200, B300,

            B38400, B19200, B9600, B4800, B2400, B1200, B300, };

int name_arr[] = {38400,  19200,  9600,  4800,  2400,  1200,  300,

            38400,  19200,  9600, 4800, 2400, 1200,  300, };

void set_speed(int fd, int speed)

{

  int   i;

  int   status;

  struct termios   Opt;

  tcgetattr(fd, &Opt);

  for ( i= 0;  i < sizeof(speed_arr) / sizeof(int);  i++)

   {

        if  (speed == name_arr[i])

        {

            tcflush(fd, TCIOFLUSH);

        cfsetispeed(&Opt, speed_arr[i]);

        cfsetospeed(&Opt, speed_arr[i]);

        status = tcsetattr(fd, TCSANOW, &Opt);

        if  (status != 0)

            perror("tcsetattr fd1");

        return;

        }

   tcflush(fd,TCIOFLUSH);

   }

}

/**

*@brief   設置串口資料位元，停止位元和效驗位

*@param  fd     類型  int  打開的串口文件控制碼*

*@param  databits 類型  int 資料位元   取值 為 7 或者8*

*@param  stopbits 類型  int 停止位   取值為 1 或者2*

*@param  parity  類型  int  效驗類型 取值為N,E,O,,S

*/

int set_Parity(int fd,int databits,int stopbits,int parity)

{

        struct termios options;

 if  ( tcgetattr( fd,&options)  !=  0)

  {

        perror("SetupSerial 1");

        return(false);

  }

  options.c_cflag &= ~CSIZE;

  switch (databits) /*設置數據位元數*/

  {

        case 7:

                 options.c_cflag |= CS7;

                 break;

        case 8:

                 options.c_cflag |= CS8;

                 break;

        default:

                 fprintf(stderr,"Unsupported data size\n");

                 return (false);

        }

  switch (parity)

        {

        case 'n':

        case 'N':

                 options.c_cflag &= ~PARENB;   /* Clear parity enable */

                 options.c_iflag &= ~INPCK;     /* Enable parity checking */

                 break;

        case 'o':

        case 'O':

                 options.c_cflag |= (PARODD | PARENB);  /* 設置為奇效驗*/ 

                 options.c_iflag |= INPCK;             /* Disnable parity checking */

                 break;

        case 'e':

        case 'E':

                 options.c_cflag |= PARENB;     /* Enable parity */

                 options.c_cflag &= ~PARODD;   /* 轉換為偶效驗*/  

                 options.c_iflag |= INPCK;       /* Disnable parity checking */

                 break;

        case 'S':

        case 's':  /*as no parity*/

                 options.c_cflag &= ~PARENB;

                 options.c_cflag &= ~CSTOPB;

                 break;

        default:

                 fprintf(stderr,"Unsupported parity\n");

                 return (false);

                 }

  /* 設置停止位*/   

  switch (stopbits)

        {

        case 1:

                 options.c_cflag &= ~CSTOPB;

                 break;

        case 2:

                 options.c_cflag |= CSTOPB;

                 break;

        default:

                 fprintf(stderr,"Unsupported stop bits\n");

                 return (false);

        }

  /* Set input parity option */

  if (parity != 'n')

                 options.c_iflag |= INPCK;

    options.c_cc[VTIME] = 150; // 15 seconds

    options.c_cc[VMIN] = 0;

    options.c_cc[VINTR] = 0;
 
    //options.c_cflag = CLOCAL|CREAD;
    //options.c_iflag = IGNPAR;
    //options.c_lflag |= ~(ICANON|ECHO|ECHOE|ISIG);
    //options.c_oflag &= ~OPOST;
    //options.c_iflag &= ~(INLCR);

  tcflush(fd,TCIFLUSH); /* Update the options and do it NOW */

  if (tcsetattr(fd,TCSANOW,&options) != 0)

        {

                 perror("SetupSerial 3");

                 return (false);

        }

  return (true);

 }

/**

*@breif 打開串口

*/

int OpenDev(char *Dev)

{

int     fd = open( Dev, O_RDWR );         //| O_NOCTTY | O_NDELAY

        if (-1 == fd)

                 { /*設置數據位元數*/

                         perror("Can't Open Serial Port");

                         return -1;

                 }

        else

        return fd;

 

}

/**

*@breif  main()

*/

int main(int argc, char **argv)

{
        int res;

        int fd;

        int nread;

        char buff[512];

        char *dev =argv[1];

        char ch;

        static char s1[20];

        fd = OpenDev(dev);

        if (fd>0)

    set_speed(fd,9600);

        else

                 {

                 printf("Can't Open Serial Port!\n");

                 exit(0);

                 }

  if (set_Parity(fd,8,1,'N')== false)

  {

    printf("Set Parity Error\n");

    exit(1);

  }
  printf("Write...\n");

  char str1[]={'\x02','\x30','\x33','\x34','\x30','\x43','\x39','\x03'};
  char str2[]={'\x02','\x30','\x33','\x34','\x31','\x31','\x46','\x42','\x03','\n'};
  char str3[]={'\x02','\x30','\x33','\x34','\x37','\x30','\x31','\x52','\x30','\x32','\x39','\x36','\x37','\x30','\x34','\x30','\x31','\x35','\x30','\x03'};
  char str4[]={'\x02','\x30','\x33','\x34','\x36','\x30','\x31','\x52','\x30','\x32','\x39','\x36','\x36','\x38','\x39','\x03','\n'};
  //res=write(fd,str2,sizeof(str2));
  //usleep(5000000);
  //printf("Run\n");
      while(1){ 
            res=write(fd,str3,sizeof(str3));
            printf("write:%d\n",res);
            usleep(1000000);
        }

    //close(fd);

    //exit(0);

}