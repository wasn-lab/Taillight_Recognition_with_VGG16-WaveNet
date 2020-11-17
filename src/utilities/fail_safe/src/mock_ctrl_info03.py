#!/usr/bin/env python
import argparse
import rospy
from msgs.msg import Flag_Info

class MockCtrlInfo03Generator():
    """Use to generate fake ctrl_info03"""
    def __init__(self, enable_aeb, enable_acc, enable_xbywire, fps):
        rospy.init_node("MockCtrlInfo03Generator")
        self.enable_aeb = enable_aeb
        self.enable_acc = enable_acc
        self.enable_xbywire = enable_xbywire
        self.fps = fps;
        self.publisher = rospy.Publisher("/Flag_Info03", Flag_Info, queue_size=1)

    def run(self):
        rate = rospy.Rate(self.fps)
        cnt = 0
        while not rospy.is_shutdown():
            if cnt == 0:
                rospy.logwarn("publish mock Flag_Info03")
            cnt += 1
            if cnt == 100:
                cnt = 0
            msg = Flag_Info()
            msg.Dspace_Flag06 = 1.0 if self.enable_xbywire else 0
            msg.Dspace_Flag07 = 1.0 if self.enable_aeb else 0
            msg.Dspace_Flag08 = 1.0 if self.enable_acc else 0
            self.publisher.publish(msg)
            rate.sleep()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-aeb", action="store_true")
    parser.add_argument("--enable-acc", action="store_true")
    parser.add_argument("--enable-xbywire", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_known_args()[0]
    gnr = MockCtrlInfo03Generator(args.enable_aeb, args.enable_acc, args.enable_xbywire, args.fps)
    gnr.run()


if __name__ == "__main__":
    main()
