/*
 * UdpClientServer.h
 *
 *  Created on: 2017年5月10日
 *      Author: user
 */

#ifndef UDPCLIENTSERVER_H_
#define UDPCLIENTSERVER_H_

#include "json.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <functional>
#include <stdexcept>
#include <sys/types.h>
#include <sys/socket.h>
#include <cerrno>
#include <unistd.h>  //sleep
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ctime>
#include <ctime>
#include <climits>
#include <cmath>
#include <vector>
#include <pcap.h>
using json = nlohmann::json;

class udp_client_server_runtime_error : public std::runtime_error
{
public:
  udp_client_server_runtime_error(const char* w) : std::runtime_error(w)
  {
  }
};

class UdpClient
{
public:
  UdpClient();
  UdpClient(const std::string& addr, int port);
  ~UdpClient();

  void initial(const std::string& addr, int port);

  int get_socket() const;
  int get_port() const;
  std::string get_addr() const;

  int send(const char* msg, size_t size);
  int send_obj_to_server(const std::string& str, bool show);

private:
  int f_socket;
  int f_port;
  std::string f_addr;
  struct addrinfo* f_addrinfo;
};

class UdpServer
{
public:
  UdpServer(const std::string& addr, int port);
  ~UdpServer();

  int get_socket() const;
  int get_port() const;
  std::string get_addr() const;

  int recv(char* msg, size_t max_size);
  int timed_recv(char* msg, size_t max_size, int max_wait_ms);

private:
  int f_port;
  std::string f_addr;
  int f_socket;
  struct addrinfo* f_addrinfo;
};

#endif /* UDPCLIENTSERVER_H_ */
