#ifndef UDPCLIENTSERVER_H_
#define UDPCLIENTSERVER_H_

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netdb.h>

#include "json.hpp"

#include "../UserDefine.h"

using namespace std;
using json = nlohmann::json;

class udp_client_server_runtime_error : public std::runtime_error
{
  public:
    udp_client_server_runtime_error (const char *w) :
        std::runtime_error (w)
    {
    }
};

class UdpClient
{
  public:
    UdpClient ();
    UdpClient (const std::string& addr,
               int port);
    ~UdpClient ();

    void
    initial (const std::string& addr,
             int port);

    int
    get_socket () const;
    int
    get_port () const;
    std::string
    get_addr () const;

    int
    send (const char *msg,
          size_t size);
    int
    send_obj_to_vcu (CLUSTER_INFO* cluster_info,
                     int cluster_size);

    int
    send_obj_to_server (CLUSTER_INFO* cluster_info,
                        int cluster_size);
    int
    send_obj_to_rsu (CLUSTER_INFO* cluster_info,
                     int cluster_size);

  private:
    int f_socket;
    int f_port;
    std::string f_addr;
    struct addrinfo * f_addrinfo;
};

class UdpServer
{
  public:
    UdpServer (const std::string& addr,
               int port);
    ~UdpServer ();

    int
    get_socket () const;
    int
    get_port () const;
    std::string
    get_addr () const;

    int
    recv (char *msg,
          size_t max_size);
    int
    timed_recv (char *msg,
                size_t max_size,
                int max_wait_ms);

  private:
    int f_port;
    string f_addr;
    int f_socket;
    struct addrinfo * f_addrinfo;
};

#endif /* UDPCLIENTSERVER_H_ */
