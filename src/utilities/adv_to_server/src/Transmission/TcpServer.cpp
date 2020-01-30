#include "TcpServer.h"
#include <unistd.h>
#include "RosModule.hpp"
using namespace std;

TcpServer::TcpServer()
{
}

void TcpServer::initial(std::string ip, int port)
{
  // create socket
  if ((listen_socket_fd_int_ = socket(AF_INET, SOCK_STREAM, 0)) == 0)
  {
    perror("socket failed");
    exit(EXIT_FAILURE);
  }
  // reuse port
  if (setsockopt(listen_socket_fd_int_, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)))
  {
    perror("setsockopt failed");
    exit(EXIT_FAILURE);
  }
  // set ADDRESS FAMILY
  server_address_struct_.sin_family = AF_INET;
  // set ip
  server_address_struct_.sin_addr.s_addr = inet_addr(ip.c_str());
  // set port
  server_address_struct_.sin_port = htons(port);

  // bind socket and socket info （family, ip, port）
  if (bind(listen_socket_fd_int_, (struct sockaddr*)&server_address_struct_, sizeof(server_address_struct_)) < 0)
  {
    perror("bind failed");
    exit(EXIT_FAILURE);
  }
  else
  {
    is_bound = true;
  }
}

TcpServer::~TcpServer()
{
  // clear address info and close socket
  // freeaddrinfo((struct addrinfo *) &server_address_struct_);
  close(listen_socket_fd_int_);
}

// get socket id for listening.
int TcpServer::get_socket() const
{
  return listen_socket_fd_int_;
}

// start listing,max connect request queue size :MAX_CONNECTION
int TcpServer::start_listening()
{
  int result = -1;
  cout << "is_bound: " << is_bound << endl;
  if (is_bound)
  {
    result = listen(listen_socket_fd_int_, MAX_CONNECTION);
  }
  if (result >= 0)
  {
    is_listening = true;
  }
  return result;
}

// wait connect request and accept and read request data
int TcpServer::wait_and_accept(void (*cb)(string))
{
  while (true)
  {
    if (is_listening)
    {
      cout << "listen_socket_fd_int_ " << listen_socket_fd_int_ << endl;

      client_socket_fd_int_ =
          accept(listen_socket_fd_int_, (struct sockaddr*)&client_address_struct_, (socklen_t*)&client_addrlen_int_);

      cout << "client fd: " << client_socket_fd_int_ << endl;
      cout << "client ip address: " << inet_ntoa(client_address_struct_.sin_addr) << endl;
      cout << "client port: " << (int)ntohs(client_address_struct_.sin_port) << endl;
      if (client_socket_fd_int_ >= 0)
      {
        cout << "start receive data" << endl;
        struct timeval tv;
        tv.tv_sec = 10;
        tv.tv_usec = 0;
        setsockopt(client_socket_fd_int_, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

        auto t1 = std::chrono::high_resolution_clock::now();
        int rel = recv(client_socket_fd_int_, buffer, sizeof(buffer), 0);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "read duration " << duration << std::endl;
        if (rel < 0)
        {
          std::string errorMsg = "runtime exception : recv data Failed. rel: " + std::to_string(rel);
          throw tcp_server_runtime_error(errorMsg);
        }

        cout << "buffer: " << buffer << endl;
        string request = buffer;
        // reset buffer every time.
        memset(&buffer, 0, sizeof(buffer));
        cb(request);
      }
    }
  }
  return 0;
}

// send data
int TcpServer::send_json(std::string json)
{
  cout << "json: " << json << endl;
  cout << "send with client fd " << client_socket_fd_int_ << std::endl;
  int rel;
  rel = send(client_socket_fd_int_, json.c_str(), json.size(), MSG_CONFIRM);
  std::cout << "rel " << rel << std::endl;
  close(client_socket_fd_int_);
  return rel;
}
