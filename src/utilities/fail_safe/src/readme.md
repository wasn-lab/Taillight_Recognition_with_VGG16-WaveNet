### Fail-safe system

The purpose of the fail-safe system is to ensure the safty when the car is running.
It keeps reporting the status of the car to backend and the pad in the driver set.
When unexpected events happen, it may disable some self-driving features and
driver will know it and take appropriate measures. 

It uses ROS messages and MQTT to inform backend and the pad. The messages include
- Status of the car.
  The status is a collection of events and states of all the components, such as sensors, detections, controls etc.
- Sensor status
  This is regulated by the govenment. The messages are sent to backend first.
  The backend server then pushed them to a platform that govern all self-driving cars.

In addition, the fail-safe system automatically upload event bags to the backend.
When unexpected events happen, it notifies message recorder to backup the current rosbag,
and then upload the rosbags to the backend.
To enable this feature, please set `LFTP_PASSWORD` and [rosbag_sender.ini](rosbag_sender.ini) correctly.




