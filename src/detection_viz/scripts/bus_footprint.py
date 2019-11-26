#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PolygonStamped
from geometry_msgs.msg import Point32

def talker():
    pub = rospy.Publisher('bus_footprint', PolygonStamped, queue_size=10)
    rospy.init_node('bus_footprint', anonymous=True)
    #
    msg = PolygonStamped()
    msg.header.stamp = rospy.get_rostime()
    msg.header.frame_id = "base_footprint"
    points = [Point32() for _ in range(4)]
    points[0].x = 3.5
    points[0].y = 1.2
    points[1].x = 3.5
    points[1].y = -1.2
    points[2].x = -3.5
    points[2].y = -1.2
    points[3].x = -3.5
    points[3].y = 1.2
    msg.polygon.points = points
    #
    rate = rospy.Rate(30) # 10hz
    while not rospy.is_shutdown():
        msg.header.stamp = rospy.get_rostime()
        pub.publish(msg)
        try:
            rate.sleep()
        except:
            pass

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
