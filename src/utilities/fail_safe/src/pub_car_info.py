#!/usr/bin/env python
# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
from __future__ import print_function
import json
import rospy
from std_msgs.msg import String

def get_car_info():
    return {
        "car_model": rospy.get_param(
            "/car_model",
            "undef_car_model"),
        "licence_plate_number": rospy.get_param(
            "/south_bridge/license_plate_number",
            "undef_license_plate_number"),
        "vid": rospy.get_param(
            "/south_bridge/vid",
            "undef_vid"),
        "company_name": rospy.get_param(
            "/south_bridge/company_name",
            "undef_company")}


def main():
    # ROS topic/node name does not allow characters like -
    node_name = "CarInfoNode"
    rospy.init_node(node_name)
    rospy.logwarn("Init %s", node_name)
    rate = rospy.Rate(1)  # FPS: 1

    car_info = get_car_info()
    msg = String()
    msg.data = json.dumps(car_info)

    topic = "/vehicle/report/car_info"
    rospy.logwarn("Publish latched data on %s", topic)
    print(json.dumps(car_info, indent=2))

    publisher = rospy.Publisher(topic, String, queue_size=1, latch=True)
    publisher.publish(msg)
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == "__main__":
    main()
