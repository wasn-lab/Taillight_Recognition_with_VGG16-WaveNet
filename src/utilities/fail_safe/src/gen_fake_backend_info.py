import argparse
import rospy
from msgs.msg import BackendInfo

class FakeBackendInfoGenerator():
    """Use to generate fake backend info, esp for battery info"""
    def __init__(self, gross_voltage, lowest_voltage):
        rospy.init_node("FakeBackendInfoGenerator")
        self.gross_voltage = gross_voltage
        self.lowest_voltage = lowest_voltage
        self.publisher = rospy.Publisher("/Backend/Info", BackendInfo, queue_size=1)

    def run(self):
        rate = rospy.Rate(10)
        cnt = 0
        while not rospy.is_shutdown():
            if cnt == 0:
                rospy.logwarn("publish fake backend info")
            cnt += 1
            if cnt == 100:
                cnt = 0
            msg = BackendInfo()
            msg.gross_voltage = self.gross_voltage
            msg.lowest_volage = self.lowest_voltage
            self.publisher.publish(msg)
            rate.sleep()

def main():
    parser = argparse.ArgumentParser()
    # gross voltage:
    #    > 355: OK
    #    355 ~ 350: WARN, need to recharge
    #    < 350: FATAL, the car has to stop
    # lowest_voltage:
    #    > 3.25: OK
    #    3.25 ~ 3.2: WARN, need to recharge
    #    < 3.2: FATAL, the car has to stop
    parser.add_argument("--gross-voltage", type=float, default=356.0)
    parser.add_argument("--lowest-voltage", type=float, default=3.26)
    args = parser.parse_args()
    gnr = FakeBackendInfoGenerator(args.gross_voltage, args.lowest_voltage)
    gnr.run()


if __name__ == "__main__":
    main()
