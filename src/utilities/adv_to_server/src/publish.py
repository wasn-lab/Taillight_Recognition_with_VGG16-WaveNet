#!/usr/bin/env python
# license removed for brevity
#test tool
import rospy
import json
from std_msgs.msg import String
import time;

def talker():
    pub = rospy.Publisher('/ADV_op/event_json', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(0.2) # 10hz
    count = 0
    status = ["ok","error", "fatal"]
    event_strs = ["AEB", "ACC","env_pedcross"]
    while not rospy.is_shutdown():
        count+=1
        current_time = time.time()
        #print(current_time)
        obj = {"module":"brake_status", "status": "error", "event_str" : event_strs[count % 3], "timestamp": current_time }
        #rospy.loginfo(json)
        pub.publish(json.dumps(obj))
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
