#ifndef MQTTCLIENT_H_
#define MQTTCLIENT_H_


#define MSG_MAX_SIZE 1

#define MQTT_HOST "iot.stois.nchc.tw"

#define MQTT_KEEP_ALIVE 60
#define MQTT_TIMEOUT_MILLI 5000
/*
#define MQTT_CA_CRT "/home/roger/itriadv/src/utilities/adv_to_server/src/Transmission/TLS/ca.crt"
#define MQTT_CLIENT_CRT "/home/roger/itriadv/src/utilities/adv_to_server/src/Transmission/TLS/client.crt"
#define MQTT_CLIENT_KEY "/home/roger/itriadv/src/utilities/adv_to_server/src/Transmission/TLS/client.key"
*/

/*
#define MQTT_CA_CRT "~/itriadv/src/utilities/adv_to_server/src/Transmission/TLS/ca.crt"
#define MQTT_CLIENT_CRT "~/itriadv/src/utilities/adv_to_server/src/Transmission/TLS/client.crt"
#define MQTT_CLIENT_KEY "~/itriadv/src/utilities/adv_to_server/src/Transmission/TLS/client.key"
*/

#include <iostream>
#include "mosquitto.h"
#include <ros/package.h>


class MqttClient
{
public:
  MqttClient();
  ~MqttClient();
  int connect();
  int publish(const std::string& topic, const std::string& msg);
  int subscribe(const std::string& topic);
  std::string vid;
  void setOnConneclCallback(void (*on_connect)(struct mosquitto* , void* , int ));
private:
  void setTLS();
  void setCallback();
  struct mosquitto* client;
  std::string broker_host;
  int broker_port;
  int broker_keep_alive;
  std::string MQTT_CA_CRT;
  std::string MQTT_CLIENT_CRT;
  std::string MQTT_CLIENT_KEY;
  int port;
};

#endif  // MQTTCLIENT_H_
