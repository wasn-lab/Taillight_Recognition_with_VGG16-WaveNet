#include "MqttClient.h"

static void on_disconnect(struct mosquitto* client, void* obj, int rc)
{
  std::string result = ": disconnect is unexpected";
  if (rc == 0)
  {
    result = ": client has called mosquitto_disconnect";
  }
  std::cout << "on_disconnect result= " << rc << result << std::endl;
}

static void on_publish(struct mosquitto* client, void* obj, int mid)
{
  std::cout << "on_publish success" << std::endl;
}

static void on_subscribe(struct mosquitto* client, void* obj, int mid, int qos_count, const int* granted_qos)
{
  std::cout << "subscribe success 123" << std::endl;
}

MqttClient::MqttClient()
{
  int rc = mosquitto_lib_init();
  client = mosquitto_new(NULL, true, NULL);
  setTLS();
  setCallback();
}

MqttClient::~MqttClient()
{
  mosquitto_disconnect(client);
  mosquitto_destroy(client);
  mosquitto_lib_cleanup();
}

int MqttClient::connect()
{
  int rc = mosquitto_connect(client, MQTT_HOST, MQTT_PORT, MQTT_KEEP_ALIVE);
  std::cout << "connect rc= " << rc << std::endl;
  mosquitto_loop_forever(client, MQTT_TIMEOUT_MILLI, MSG_MAX_SIZE);
  return rc;
}

int MqttClient::publish(const std::string& topic, const std::string& msg)
{
  int* msg_id = NULL;
  const char* topic_name = topic.c_str();
  size_t payload_length = msg.length();
  const char* payload = msg.c_str();
  int qos = 0;
  bool retain = false;
  int rc =  mosquitto_publish(client, msg_id, topic_name, payload_length, payload, qos, retain);
  std::cout << "publish rc= " << rc << std::endl;
  return rc;
}

int MqttClient::subscribe(const std::string& topic)
{
  std::cout << "to do" << std::endl;
  return 1;
}

void MqttClient::setTLS()
{
  int rc = mosquitto_tls_opts_set(client, 1, "tlsv1.1", NULL);
  std::cout << "mosquitto_tls_opts_set rc: " << rc << std::endl;
  std::string path = ros::package::getPath("adv_to_server");
  MQTT_CA_CRT = path + "/src/Transmission/TLS/ca.crt";
  MQTT_CLIENT_CRT = path + "/src/Transmission/TLS/client.crt";
  MQTT_CLIENT_KEY = path + "/src/Transmission/TLS/client.key";
  rc = mosquitto_tls_set(client, MQTT_CA_CRT.c_str(), NULL, MQTT_CLIENT_CRT.c_str(), MQTT_CLIENT_KEY.c_str(), 0);
  std::cout << "mosquitto_tls_set rc: " << rc << std::endl;
  rc = mosquitto_tls_insecure_set(client, false);
  std::cout << "mosquitto_tls_insecure_set rc: " << rc << std::endl;
}

void MqttClient::setCallback()
{

  mosquitto_disconnect_callback_set(client, on_disconnect);
  mosquitto_publish_callback_set(client, on_publish);
  mosquitto_subscribe_callback_set(client, on_subscribe);
}

void MqttClient::setOnConneclCallback( void (*on_connect)(struct mosquitto* , void* , int ))
{
  mosquitto_connect_callback_set(client, on_connect);
}
