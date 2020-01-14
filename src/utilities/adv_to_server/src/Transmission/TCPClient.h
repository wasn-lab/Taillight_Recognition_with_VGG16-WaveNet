#ifndef TCPCLIENT_H_
#define TCPCLIENT_H_
#include "json.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <functional>
#include <stdexcept>
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include <unistd.h>  //sleep
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <ctime>
#include <limits.h>
#include <math.h>
#include <vector>
#include <pcap.h>

class tcp_client_runtime_error : public std::runtime_error
{
public:
  tcp_client_runtime_error(const char* w) : std::runtime_error(w)
  {
  }
};

class TCPClient
{
public:
  TCPClient();

  ~TCPClient();

  void initial(const std::string& addr, int port);


  int sendRequest(const char* msg, size_t size);
  int recvResponse(char buffer[2048], size_t size);

  int connectServer();

private:
  int f_socket;
  int f_port;
  std::string f_addr;
  struct addrinfo* f_addrinfo;
};
#endif
