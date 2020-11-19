import unittest
import time
from itri_mqtt_client import ItriMqttClient

CLIENT = ItriMqttClient("127.0.0.1")

class ItriMqttClientTest(unittest.TestCase):
    def test_1(self):
        for i in range(60):
            CLIENT.publish("/mqtt_test", "kerker {}".format(i))
            time.sleep(1)


if __name__ == "__main__":
    unittest.main()
