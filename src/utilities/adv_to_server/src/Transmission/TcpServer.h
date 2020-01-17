/*
 * TCPServer.h
 *
 *  Created on: 2019年12月06日
 *      Author: Roger Chen
 */

#ifndef TCPSERVER_H_
#define TCPSERVER_H_

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
#include "ros/ros.h"

#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Int32MultiArray.h"
#include "std_msgs/String.h"

#define PORT 8080
#define MAX_CONNECTION 1
using json = nlohmann::json;


class tcp_server_runtime_error : public std::runtime_error
{
public:
  tcp_server_runtime_error(std::string w) : std::runtime_error(w)
  {
  }
};


class TcpServer
{
public:
  //建構子
  TcpServer();
  void initial(std::string ip, int port);
  //解構子
  ~TcpServer();
  //取得client socket描述符
  int get_socket() const;

  //開始監聽
  int start_listening();
  //等待並讀取資料
  int wait_and_accept(void (*cb)(std::string));
  //回覆資料
  int send_json(std::string json);

private:
  void handleRequest(std::string request);
  //是否正在監聽連線
  bool is_listening = false;
  //是否綁定位址
  bool is_bound = false;
  //監聽socket的描述符
  int listen_socket_fd_int_;
  // client端socket的描述符
  int client_socket_fd_int_;
  // socket 資料結構
  struct sockaddr_in server_address_struct_;
  struct sockaddr_in client_address_struct_;
  std::string request;
  // option
  int opt = 1;
  // ip 長度
  int client_addrlen_int_ = sizeof(client_address_struct_);

  // buffer
  char buffer[1024] = { 0 };
};

#endif
