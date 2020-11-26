#ifndef MQTTCLIENT_H_
#define MQTTCLIENT_H_


#define MSG_MAX_SIZE 1

#define MQTT_HOST "iot.stois.nchc.tw"
#define MQTT_PORT 3053
#define MQTT_KEEP_ALIVE 60
#define MQTT_TIMEOUT_MILLI 5000
/*
#define MQTT_CA_CRT "/home/roger/itriadv/src/utilities/adv_to_server/src/Transmission/TLS/ca.crt"
#define MQTT_CLIENT_CRT "/home/roger/itriadv/src/utilities/adv_to_server/src/Transmission/TLS/client.crt"
#define MQTT_CLIENT_KEY "/home/roger/itriadv/src/utilities/adv_to_server/src/Transmission/TLS/client.key"
*/

#define MQTT_CA_CRT "src/utilities/adv_to_server/src/Transmission/TLS/ca.crt"
#define MQTT_CLIENT_CRT "src/utilities/adv_to_server/src/Transmission/TLS/client.crt"
#define MQTT_CLIENT_KEY "src/utilities/adv_to_server/src/Transmission/TLS/client.key"

#include <iostream>
#include "mosquitto.h"

class MqttClient
{
public:
  MqttClient();
  ~MqttClient();
  int connect();
  int publish(std::string topic, std::string msg);
  int subscribe(std::string topic);
  void setOnConneclCallback(void (*on_connect)(struct mosquitto* , void* , int ));
private:
  void setTLS();
  void setCallback();
  struct mosquitto* client;
  std::string broker_host;
  int broker_port;
  int broker_keep_alive;
};

#endif  // MQTTCLIENT_H_