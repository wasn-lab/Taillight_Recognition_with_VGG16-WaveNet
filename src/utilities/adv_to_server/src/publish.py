#!/usr/bin/env python
# license removed for brevity
# 測試工具
import rospy
import json
from std_msgs.msg import String
import time;

def talker():
    pub = rospy.Publisher('/ADV_op/event_json', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(1000) # 10hz
    count = 0
    while not rospy.is_shutdown():
        count+=1
        current_time = time.time()
        #print(current_time)
        obj = {"module":"offline test", "status": "ok", "event_str" : "AEB", "timestamp": current_time }
        #rospy.loginfo(json)
        pub.publish(json.dumps(obj))
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
