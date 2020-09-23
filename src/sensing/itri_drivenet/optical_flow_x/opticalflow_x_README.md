### Install drivenet dependencies

* Install Opencv >= 4.0
* Install python-rospy
* Install python-cv-bridge


```
sudo pip install opencv-python
sudo apt install python-rospy
sudo apt-get install ros-(ROS version name)-cv-bridge
```


### How to run
```
python2 optical_detect_20200923_ros_per5_yolo.py
```

### How to evaluate results

* debug image topic
```
rostopic echo /cam/front_bottom_60/optical_flow
```

* tracking bbox topic
```
rostopic echo /optical_flow/front_bottom_60
```
