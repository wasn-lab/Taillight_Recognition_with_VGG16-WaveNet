// Client side C/C++ program to demonstrate Socket programming
#include <cstdio>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include "TCPClient.h"
#include <chrono>
using namespace std;

TCPClient::TCPClient()
{
  f_port = -1;
  f_addr = "";
  f_socket = -1;
  f_addrinfo = NULL;
}

void TCPClient::initial(const std::string& addr, int port)
{
  f_port = port;
  f_addr = addr;

  addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;

  int result = getaddrinfo(f_addr.c_str(), to_string(f_port).c_str(), &hints, &f_addrinfo);

  if (result != 0 || f_addrinfo == NULL)
  {
    throw tcp_client_runtime_error(
        ("runtime exception : invalid address or port: \"" + addr + ":" + to_string(port) + "\"").c_str());
  }

  f_socket = socket(f_addrinfo->ai_family, SOCK_STREAM, IPPROTO_TCP);

  if (f_socket == -1)
  {
    freeaddrinfo(f_addrinfo);
    throw tcp_client_runtime_error(
        ("runtime exception : could not create socket for: \"" + addr + ":" + to_string(port) + "\"").c_str());
  }
}

int TCPClient::connectServer()
{
  struct timeval timeout;
  timeout.tv_sec = 3;
  timeout.tv_usec = 0;

  setsockopt(f_socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  // auto t1 = std::chrono::high_resolution_clock::now();
  int rel = connect(f_socket, f_addrinfo->ai_addr, f_addrinfo->ai_addrlen);
  // auto t2 = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  // std::cout << "connect duration " << duration << std::endl;
  std::cout << "connect result: " << rel << std::endl;
  if (rel < 0)
  {
    std::string errorMsg = "runtime exception : Connect failed. rel: " + std::to_string(rel);
    throw tcp_client_runtime_error(errorMsg);
  }
  return rel;
}

int TCPClient::sendRequest(const char* msg, size_t size)
{
  std::string json(msg);
  int rel = send(f_socket, msg, size, 0);
  if (rel < 0)
  {
    std::string errorMsg = "runtime exception : send json Failed. rel: " + std::to_string(rel);
    throw tcp_client_runtime_error(errorMsg);
  }
  return rel;
}

int TCPClient::recvResponse(char buffer[2048], size_t size)
{
  struct timeval tv;
  tv.tv_sec = 5;
  tv.tv_usec = 0;
  setsockopt(f_socket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

  // auto t1 = std::chrono::high_resolution_clock::now();
  int rel = read(f_socket, buffer, size);
  // auto t2 = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
  // std::cout << "read duration " << duration << std::endl;
  if (rel < 0)
  {
    std::string errorMsg = "runtime exception : read data Failed. rel: " + std::to_string(rel);
    throw tcp_client_runtime_error(errorMsg);
  }
  // std::cout << "tcpClient receive: " << buffer << std::endl;
  return rel;
}

TCPClient::~TCPClient()
{
  freeaddrinfo(f_addrinfo);
  close(f_socket);
}
