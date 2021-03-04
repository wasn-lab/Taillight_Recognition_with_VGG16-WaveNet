# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
"""
Aggregate cpu/gpu loads and publish it
"""
import datetime
import json
import rospy
from std_msgs.msg import String
from car_model_helper import get_car_model_as_str


class LoadCollector(object):
    def __init__(self):
        rospy.init_node("LoadCollector")
        car_model = get_car_model_as_str()
        self.ipcs = []
        if car_model in ["B1", "B1_V2", "B1_V3", "C1"]:
            self.ipcs = ["lidar", "camera", "localization", "xavier"]
        else:
            rospy.logerr("Unrecognized car_model: %s", car_model)

        rospy.logwarn("Init LoadCollector node")
        self.records, self.timestamps = self.setup_records()

        for ipc in self.ipcs:
            cpu_topic = "/vehicle/report/{}/cpu_load".format(ipc)
            rospy.Subscriber(cpu_topic, String, callback=self._cb, callback_args=("cpu_load", ipc), queue_size=1)
            rospy.logwarn("Subscribe %s", cpu_topic)
            gpu_topic = "/vehicle/report/{}/gpu_load".format(ipc)
            rospy.Subscriber(gpu_topic, String, callback=self._cb, callback_args=("gpu_load", ipc), queue_size=1)
            rospy.logwarn("Subscribe %s", gpu_topic)

        output_topic = "/vehicle/report/system_loads"
        self.load_publisher = rospy.Publisher(output_topic, String, queue_size=10)
        rospy.logwarn("Publish on %s", output_topic)

    def setup_records(self):
        now = datetime.datetime.now()
        self.records = {}
        self.timestamps = {}
        for ipc in self.ipcs:
            self.records[ipc] = {"cpu_load": "", "gpu_load": ""}
            self.timestamps[ipc] = {"cpu_load": now, "gpu_load": now}
        return self.records, self.timestamps

    def _cb(self, msg, args):
        load_type, ipc_name = args
        self.records[ipc_name][load_type] = msg.data
        self.timestamps[ipc_name][load_type] = datetime.datetime.now()

    def get_current_loads(self, now=None):
        """
        Return the latest loads.
        If we do not get cpu/gpu loads in recent 3 seconds, then show it as NA
        """
        if now is None:
            now = datetime.datetime.now()
        ret = {}
        for ipc in self.ipcs:
            ret[ipc] = {}
            for load_type in ["cpu_load", "gpu_load"]:
                delta = now - self.timestamps[ipc][load_type]
                if delta.total_seconds() > 3:
                    ret[ipc][load_type] = "NA"
                else:
                    ret[ipc][load_type] = self.records[ipc][load_type]
        return ret

    def run(self):
        rate = rospy.Rate(1)  # FPS: 1
        while not rospy.is_shutdown():
            loads = self.get_current_loads()
            self.load_publisher.publish(json.dumps(loads))
            rate.sleep()

if __name__ == "__main__":
    col = LoadCollector()
    col.run()
