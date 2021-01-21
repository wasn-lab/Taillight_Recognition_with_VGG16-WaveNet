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
        if car_model in ["B1_V2", "B1_V3", "C1"]:
            self.ipcs = ["lidar", "camera", "localization", "xavier"]
        else:
            rospy.logerr("Unrecognized car_model: %s", car_model)

        rospy.logwarn("Init LoadCollector node")
        self.records = {}
        now = datetime.datetime.now()
        for ipc in self.ipcs:
            self.records[ipc] = {"cpu_load": "",
                                 "gpu_load": "",
                                 "cpu_load_ts": now,
                                 "gpu_load_ts": now}

        for ipc in self.ipcs:
            cpu_topic = "/vehicle/report/{}/cpu_load".format(ipc)
            rospy.Subscriber(cpu_topic, String, callback=self._cb, callback_args=("cpu_load", ipc), queue_size=1)
            rospy.logwarn("Subscribe %s", cpu_topic)
            gpu_topic = "/vehicle/report/{}/gpu_load".format(ipc)
            rospy.Subscriber(gpu_topic, String, callback=self._cb, callback_args=("gpu_load", ipc), queue_size=1)
            rospy.logwarn("Subscribe %s", gpu_topic)

        self.load_publisher = rospy.Publisher(
            "/vehicle/report/ipc_loads", String, queue_size=10)
        rospy.logwarn("Publish on /vehicle/report/system_loads")

    def _cb(self, msg, args):
        load_type, ipc_name = args
        self.records[ipc_name][load_type] = msg.data
        self.records[ipc_name][load_type + "_ts"] = datetime.datetime.now()

    def _evict_old_record(self):
        now = datetime.datetime.now()
        for ipc in self.ipcs:
            delta = now - self.records[ipc]["cpu_load_ts"]
            if delta.total_seconds() > 3:
                self.records[ipc]["cpu_load"] = "NA"
            delta = now - self.records[ipc]["gpu_load_ts"]
            if delta.total_seconds() > 3:
                self.records[ipc]["gpu_load"] = "NA"

    def get_current_loads(self):
        """
        Return the latest loads.
        If we do not get cpu/gpu loads in recent 3 seconds, then show it as NA
        """
        now = datetime.datetime.now()
        ret = {}
        for ipc in self.ipcs:
            ret[ipc] = {}
            for load_type in ["cpu_load", "gpu_load"]:
                ts_key = load_type + "_ts"
                delta = now - self.records[ipc][ts_key]
                if delta.total_seconds() > 3:
                    ret[ipc][load_type] = "NA"
                else:
                    ret[ipc][load_type] = self.records[ipc][load_type]
        return ret

    def run(self):
        rate = rospy.Rate(1)  # FPS: 1
        while not rospy.is_shutdown():
            cur_load = self.get_current_loads()
            self.load_publisher.publish(json.dumps(cur_load))
            rate.sleep()

if __name__ == "__main__":
    col = LoadCollector()
    col.run()
