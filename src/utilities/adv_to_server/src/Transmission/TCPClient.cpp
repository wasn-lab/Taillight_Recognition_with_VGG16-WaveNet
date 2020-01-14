// Client side C/C++ program to demonstrate Socket programming
#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include "TCPClient.h"

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
    throw tcp_client_runtime_error(("invalid address or port: \"" + addr + ":" + to_string(port) + "\"").c_str());
  }

  f_socket = socket(f_addrinfo->ai_family, SOCK_STREAM, IPPROTO_TCP);

  if (f_socket == -1)
  {
    freeaddrinfo(f_addrinfo);
    throw tcp_client_runtime_error(("could not create socket for: \"" + addr + ":" + to_string(port) + "\"").c_str());
  }
}

int TCPClient::connectServer()
{
  if (connect(f_socket, f_addrinfo->ai_addr, f_addrinfo->ai_addrlen) < 0)
  {
    printf("\nConnection Failed \n");
    return -1;
  }
  return 0;
}

int TCPClient::sendRequest(const char* msg, size_t size)
{
  std::string json(msg);
  return send(f_socket, msg, size, 0);
}

int TCPClient::recvResponse(char buffer[2048], size_t size)
{
  // memset(buffer, 0, sizeof(buffer));
  int result = read(f_socket, buffer, size);
  // std::cout << "tcpClient receive: " << buffer << std::endl;
  return result;
}

TCPClient::~TCPClient()
{
  freeaddrinfo(f_addrinfo);
  close(f_socket);
}
