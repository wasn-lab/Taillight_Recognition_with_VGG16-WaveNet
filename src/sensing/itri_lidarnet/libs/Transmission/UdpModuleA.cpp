#include "UdpModuleA.h"

// ------------------------------- client -------------------------------
UdpClient::UdpClient()
{
  f_port = -1;
  f_addr = "";
  f_socket = -1;
  f_addrinfo = NULL;
}

void UdpClient::initial(const std::string& addr, int port)
{
  f_port = port;
  f_addr = addr;

  addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;
  hints.ai_protocol = IPPROTO_UDP;

  int result = getaddrinfo(f_addr.c_str(), to_string(f_port).c_str(), &hints, &f_addrinfo);

  if (result != 0 || f_addrinfo == NULL)
  {
    throw udp_client_server_runtime_error(
        ("invalid address or port: \"" + addr + ":" + to_string(port) + "\"").c_str());
  }

  f_socket = socket(f_addrinfo->ai_family, SOCK_DGRAM | SOCK_CLOEXEC, IPPROTO_UDP);

  if (f_socket == -1)
  {
    freeaddrinfo(f_addrinfo);
    throw udp_client_server_runtime_error(
        ("could not create socket for: \"" + addr + ":" + to_string(port) + "\"").c_str());
  }
}

/** \brief Clean up the UDP client object.
 *
 * This function frees the address information structure and close the socket
 * before returning.
 */
UdpClient::~UdpClient()
{
  freeaddrinfo(f_addrinfo);
  close(f_socket);
}

/** \brief Retrieve a copy of the socket identifier.
 *
 * This function return the socket identifier as returned by the socket()
 * function. This can be used to change some flags.
 *
 * \return The socket used by this UDP client.
 */
int UdpClient::get_socket() const
{
  return f_socket;
}

/** \brief Retrieve the port used by this UDP client.
 *
 * This function returns the port used by this UDP client. The port is
 * defined as an integer, host side.
 *
 * \return The port as expected in a host integer.
 */
int UdpClient::get_port() const
{
  return f_port;
}

/** \brief Retrieve a copy of the address.
 *
 * This function returns a copy of the address as it was specified in the
 * constructor. This does not return a canonalized version of the address.
 *
 * The address cannot be modified. If you need to send data on a different
 * address, create a new UDP client.
 *
 * \return A string with a copy of the constructor input address.
 */
std::string UdpClient::get_addr() const
{
  return f_addr;
}

/** \brief Send a message through this UDP client.
 *
 * This function sends \p msg through the UDP client socket. The function
 * cannot be used to change the destination as it was defined when creating
 * the udp_client object.
 *
 * The size must be small enough for the message to fit. In most cases we
 * use these in Snap! to send very small signals (i.e. 4 bytes commands.)
 * Any data we would want to share remains in the Cassandra database so
 * that way we can avoid losing it because of a UDP message.
 *
 * \param[in] msg  The message to send.
 * \param[in] size  The number of bytes representing this message.
 *
 * \return -1 if an error occurs, otherwise the number of bytes sent. errno
 * is set accordingly on error.
 */
int UdpClient::send(const char* msg, size_t size)
{
  return sendto(f_socket, msg, size, 0, f_addrinfo->ai_addr, f_addrinfo->ai_addrlen);
}

int UdpClient::send_obj_to_vcu(CLUSTER_INFO* cluster_info, int cluster_size)
{
  if (cluster_size > 20)
  {
    cluster_size = 20;
  }

  int NUM = 25;
  char data[cluster_size * NUM];
  for (int i = 0; i < cluster_size; i++)
  {
    data[i * NUM + 0] = '$';
    data[i * NUM + 1] = NUM;
    data[i * NUM + 2] = i;
    data[i * NUM + 3] = cluster_info[i].cluster_tag;
    data[i * NUM + 4] = (int)(cluster_info[i].dis_center_origin * 100) >> 8;
    data[i * NUM + 5] = cluster_info[i].dis_center_origin;
    data[i * NUM + 6] = cluster_info[i].angle_from_x_axis;
    data[i * NUM + 7] = 100;
    data[i * NUM + 8] = (int)(cluster_info[i].obb_vertex[0].x * 100) >> 8;
    data[i * NUM + 9] = (int)(cluster_info[i].obb_vertex[0].x * 100) & 0xff;
    data[i * NUM + 10] = (int)(cluster_info[i].obb_vertex[0].y * 100) >> 8;
    data[i * NUM + 11] = (int)(cluster_info[i].obb_vertex[0].y * 100) & 0xff;
    data[i * NUM + 12] = (int)(cluster_info[i].obb_vertex[4].x * 100) >> 8;
    data[i * NUM + 13] = (int)(cluster_info[i].obb_vertex[4].x * 100) & 0xff;
    data[i * NUM + 14] = (int)(cluster_info[i].obb_vertex[4].y * 100) >> 8;
    data[i * NUM + 15] = (int)(cluster_info[i].obb_vertex[4].y * 100) & 0xff;
    data[i * NUM + 16] = (int)(cluster_info[i].obb_vertex[7].x * 100) >> 8;
    data[i * NUM + 17] = (int)(cluster_info[i].obb_vertex[7].x * 100) & 0xff;
    data[i * NUM + 18] = (int)(cluster_info[i].obb_vertex[7].y * 100) >> 8;
    data[i * NUM + 19] = (int)(cluster_info[i].obb_vertex[7].y * 100) & 0xff;
    data[i * NUM + 20] = (int)(cluster_info[i].obb_vertex[3].x * 100) >> 8;
    data[i * NUM + 21] = (int)(cluster_info[i].obb_vertex[3].x * 100) & 0xff;
    data[i * NUM + 22] = (int)(cluster_info[i].obb_vertex[3].y * 100) >> 8;
    data[i * NUM + 23] = (int)(cluster_info[i].obb_vertex[3].y * 100) & 0xff;
    data[i * NUM + 24] = '&';
  }

  return sendto(f_socket, data, sizeof(data), 0, f_addrinfo->ai_addr, f_addrinfo->ai_addrlen);
}

int UdpClient::send_obj_to_server(CLUSTER_INFO* cluster_info, int cluster_size)
{
  stringstream stream;
  stream << fixed << setprecision(2);
  for (int i = 0; i < cluster_size; i++)
  {
    stream << "$"
           << ",";
    stream << cluster_info[i].obb_vertex[0].x << ",";
    stream << cluster_info[i].obb_vertex[0].y << ",";
    stream << cluster_info[i].obb_vertex[4].x << ",";
    stream << cluster_info[i].obb_vertex[4].y << ",";
    stream << cluster_info[i].obb_vertex[7].x << ",";
    stream << cluster_info[i].obb_vertex[7].y << ",";
    stream << cluster_info[i].obb_vertex[3].x << ",";
    stream << cluster_info[i].obb_vertex[3].y << ",";
    stream << cluster_info[i].dz << ",";
    stream << cluster_info[i].angle_from_x_axis << ",";
    stream << cluster_info[i].cluster_tag << ",";
    stream << cluster_info[i].dis_center_origin << ",";
    stream << cluster_info[i].confidence << ",";
    stream << "&";
  }
  string str = stream.str();
  const char* msg = str.c_str();
  size_t msg_length = strlen(msg);

  return sendto(f_socket, msg, msg_length, 0, f_addrinfo->ai_addr, f_addrinfo->ai_addrlen);
}

int UdpClient::send_obj_to_rsu(CLUSTER_INFO* cluster_info, int cluster_size)
{
  if (cluster_size > 0)
  {
    stringstream stream;
    stream << fixed << setprecision(2);

    time_t t = time(0);  // get time now
    struct tm* now = localtime(&t);

    char buf[30];
    strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", now);
    string time_string(buf);

    json J1;
    J1["timestamp"] = time_string;
    J1["lidar_id"] = 215;
    J1["obj_no"] = cluster_size;
    J1["objects"];
    for (int i = 0; i < cluster_size; i++)
    {
      if (cluster_info[i].cluster_tag <= 3)
      {
        json J2;
        J2["id"] = to_string(cluster_info[i].tracking_id);
        J2["x_pos"] = to_string(cluster_info[i].center.x);
        J2["y_pos"] = to_string(cluster_info[i].center.y);
        J2["x_speed"] = to_string(cluster_info[i].velocity.x * 3.6);
        J2["y_speed"] = to_string(cluster_info[i].velocity.y * 3.6);
        J2["length"] = to_string(cluster_info[i].dis_max_min);
        J1["objects"] += J2;
      }
    }

    json J0;
    J0["Lidar"] = J1;

    string str = J0.dump();
    // cout<< str << endl;
    const char* msg = str.c_str();
    size_t msg_length = strlen(msg);
    return sendto(f_socket, msg, msg_length, 0, f_addrinfo->ai_addr, f_addrinfo->ai_addrlen);
  }
  else
  {
    return -1;
  }
}

// ------------------------------- server -------------------------------
/** \brief Initialize a UDP server object.
 *
 * This function initializes a UDP server object making it ready to
 * receive messages.
 *
 * The server address and port are specified in the constructor so
 * if you need to receive messages from several different addresses
 * and/or port, you'll have to create a server for each.
 *
 * The address is a string and it can represent an IPv4 or IPv6
 * address.
 *
 * Note that this function calls connect() to connect the socket
 * to the specified address. To accept data on different UDP addresses
 * and ports, multiple UDP servers must be created.
 *
 * \note
 * The socket is open in this process. If you fork() or exec() then the
 * socket will be closed by the operating system.
 *
 * \warning
 * We only make use of the first address found by getaddrinfo(). All
 * the other addresses are ignored.
 *
 * \exception udp_client_server_runtime_error
 * The udp_client_server_runtime_error exception is raised when the address
 * and port combinaison cannot be resolved or if the socket cannot be
 * opened.
 *
 * \param[in] addr  The address we receive on.
 * \param[in] port  The port we receive from.
 */

UdpServer::UdpServer(const std::string& addr, int port) : f_port(port), f_addr(addr)
{
  char decimal_port[16];
  snprintf(decimal_port, sizeof(decimal_port), "%d", f_port);
  decimal_port[sizeof(decimal_port) / sizeof(decimal_port[0]) - 1] = '\0';
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;
  hints.ai_protocol = IPPROTO_UDP;
  int r(getaddrinfo(addr.c_str(), decimal_port, &hints, &f_addrinfo));
  if (r != 0 || f_addrinfo == NULL)
  {
    throw udp_client_server_runtime_error(
        ("invalid address or port for UDP socket: \"" + addr + ":" + decimal_port + "\"").c_str());
  }
  f_socket = socket(f_addrinfo->ai_family, SOCK_DGRAM | SOCK_CLOEXEC, IPPROTO_UDP);
  if (f_socket == -1)
  {
    freeaddrinfo(f_addrinfo);
    throw udp_client_server_runtime_error(
        ("could not create UDP socket for: \"" + addr + ":" + decimal_port + "\"").c_str());
  }
  r = bind(f_socket, f_addrinfo->ai_addr, f_addrinfo->ai_addrlen);
  if (r != 0)
  {
    freeaddrinfo(f_addrinfo);
    close(f_socket);
    throw udp_client_server_runtime_error(
        ("could not bind UDP socket with: \"" + addr + ":" + decimal_port + "\"").c_str());
  }
}

/** \brief Clean up the UDP server.
 *
 * This function frees the address info structures and close the socket.
 */
UdpServer::~UdpServer()
{
  freeaddrinfo(f_addrinfo);
  close(f_socket);
}

/** \brief The socket used by this UDP server.
 *
 * This function returns the socket identifier. It can be useful if you are
 * doing a select() on many sockets.
 *
 * \return The socket of this UDP server.
 */
int UdpServer::get_socket() const
{
  return f_socket;
}

/** \brief The port used by this UDP server.
 *
 * This function returns the port attached to the UDP server. It is a copy
 * of the port specified in the constructor.
 *
 * \return The port of the UDP server.
 */
int UdpServer::get_port() const
{
  return f_port;
}

/** \brief Return the address of this UDP server.
 *
 * This function returns a verbatim copy of the address as passed to the
 * constructor of the UDP server (i.e. it does not return the canonalized
 * version of the address.)
 *
 * \return The address as passed to the constructor.
 */
std::string UdpServer::get_addr() const
{
  return f_addr;
}

/** \brief Wait on a message.
 *
 * This function waits until a message is received on this UDP server.
 * There are no means to return from this function except by receiving
 * a message. Remember that UDP does not have a connect state so whether
 * another process quits does not change the status of this UDP server
 * and thus it continues to wait forever.
 *
 * Note that you may change the type of socket by making it non-blocking
 * (use the get_socket() to retrieve the socket identifier) in which
 * case this function will not block if no message is available. Instead
 * it returns immediately.
 *
 * \param[in] msg  The buffer where the message is saved.
 * \param[in] max_size  The maximum size the message (i.e. size of the \p msg buffer.)
 *
 * \return The number of bytes read or -1 if an error occurs.
 */
int UdpServer::recv(char* msg, size_t max_size)
{
  return ::recv(f_socket, msg, max_size, 0);
}

/** \brief Wait for data to come in.
 *
 * This function waits for a given amount of time for data to come in. If
 * no data comes in after max_wait_ms, the function returns with -1 and
 * errno set to EAGAIN.
 *
 * The socket is expected to be a blocking socket (the default,) although
 * it is possible to setup the socket as non-blocking if necessary for
 * some other reason.
 *
 * This function blocks for a maximum amount of time as defined by
 * max_wait_ms. It may return sooner with an error or a message.
 *
 * \param[in] msg  The buffer where the message will be saved.
 * \param[in] max_size  The size of the \p msg buffer in bytes.
 * \param[in] max_wait_ms  The maximum number of milliseconds to wait for a message.
 *
 * \return -1 if an error occurs or the function timed out, the number of bytes received otherwise.
 */
int UdpServer::timed_recv(char* msg, size_t max_size, int max_wait_ms)
{
  fd_set s;
  FD_ZERO(&s);
  FD_SET(f_socket, &s);
  struct timeval timeout;
  timeout.tv_sec = max_wait_ms / 1000;
  timeout.tv_usec = (max_wait_ms % 1000) * 1000;
  int retval = select(f_socket + 1, &s, &s, &s, &timeout);
  if (retval == -1)
  {
    // select() set errno accordingly
    return -1;
  }
  if (retval > 0)
  {
    // our socket has data
    return ::recv(f_socket, msg, max_size, 0);
  }

  // our socket has no data
  errno = EAGAIN;
  return -1;
}

// vim: ts=4 sw=4 et

/*
 int
 udp ()
 {

 int fd;
 if ( (fd = socket (AF_INET, SOCK_DGRAM, 0)) < 0)
 {
 perror ("socket failed!");
 return 0;
 }

 struct sockaddr_in myaddr;  address that client uses
 bzero ((char *) &myaddr, sizeof (myaddr));
 myaddr.sin_family = AF_INET;
 myaddr.sin_addr.s_addr = htonl (INADDR_ANY);
 myaddr.sin_port = htons (0);

 if (bind (fd, (struct sockaddr *) &myaddr, sizeof (myaddr)) < 0)
 {
 perror ("bind failed!");
 return 0;
 }

 struct sockaddr_in servaddr;  the server's full addr
 bzero ((char *) &servaddr, sizeof (servaddr));
 servaddr.sin_family = AF_INET;
 servaddr.sin_port = htons (8888);  //port
 struct hostent *hp;  holds IP address of server
 hp = gethostbyname ("192.168.0.2");
 if (hp == 0)
 {
 return (-1);
 }
 bcopy (hp->h_addr_list[0], (caddr_t) &servaddr.sin_addr, hp->h_length);

 char data[25] = "$PID";
 data[0] = 36;
 data[1] = 25;
 data[2] = 77;
 data[3] = 118;
 data[4] = 30;
 data[5] = 97;
 data[6] = 97;

 int size = sizeof (servaddr);
 if (sendto (fd, data, 25, 0, (struct sockaddr*) &servaddr, size) == -1)
 {
 perror ("write to server error !");
 return 0;
 }

 return 1;
 }
 */
