import rospy
from std_msgs.msg import String


class ActionEmitter(object):
    def __init__(self):
        self.recorder_backup_pub = rospy.Publisher("/REC/req_backup", String, queue_size=1000)
        rospy.logwarn("Init publisher for /REC/req_backup")

    def backup_rosbag(self, reason="unknown"):
        nlistener = self.recorder_backup_pub.get_num_connections()
        if nlistener == 0:
            rospy.logwarn("No listener for /REC/req_backup, possibly no backup is performed")
        self.recorder_backup_pub.publish(reason)
