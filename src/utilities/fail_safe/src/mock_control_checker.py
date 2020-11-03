import argparse
import rospy
from std_msgs.msg import Int8MultiArray

class MockControlCheckerGenerator():
    """Use to generate fake control checker message"""
    def __init__(self, enable_can, fps):
        rospy.init_node("MockControlCheckerGenerator")
        self.enable_can = enable_can
        self.fps = fps;
        self.publisher = rospy.Publisher("/control_checker", Int8MultiArray, queue_size=1)

    def run(self):
        rate = rospy.Rate(self.fps)
        cnt = 0
        while not rospy.is_shutdown():
            if cnt == 0:
                rospy.logwarn("publish mock /control_checker")
            cnt += 1
            if cnt == 100:
                cnt = 0
            msg = Int8MultiArray()
            if self.enable_can:
                msg.data = [0] * 9
            else:
                msg.data = [1] * 9  # total 9 elements

            self.publisher.publish(msg)
            rate.sleep()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-can", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    gnr = MockControlCheckerGenerator(args.enable_can, args.fps)
    gnr.run()


if __name__ == "__main__":
    main()
