# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import datetime
import json
import unittest
from load_collector import LoadCollector
from status_level import OK, WARN
from std_msgs.msg import String

class LoadCollectorTest(unittest.TestCase):
    def test_get_current_loads(self):
        now = datetime.datetime.now()

        collector = LoadCollector("127.0.0.1", 1883)
        collector.ipcs = ["xavier"]
        collector.setup_records()
        collector.records["xavier"]["cpu_load"] = 0.16
        collector.records["xavier"]["gpu_load"] = "47"
        collector.records["xavier"]["cpu_load_threshold"] = 12
        collector.timestamps["xavier"] = now

        delta = datetime.timedelta(seconds=1)
        for _ in range(3):
            collector.timestamps["xavier"] -= delta
            loads = collector.get_current_loads(now=now)
            self.assertEqual(loads["xavier"]["status"], OK)
            self.assertEqual(loads["xavier"]["status_str"], "")
        collector.timestamps["xavier"] -= delta * 10
        loads = collector.get_current_loads(now=now)
        self.assertEqual(loads["xavier"]["status"], WARN)

    def test_get_current_loads_high_cpu(self):
        now = datetime.datetime.now()

        collector = LoadCollector("127.0.0.1", 1883)
        collector.ipcs = ["camera"]
        collector.setup_records()

        ipc_load = {
            "hostname": "camera",
            "cpu_load": 23.14,
            "cpu_load_threshold": 18,
            "gpu_power_draw": 250,
            "gpu_temperature": 80,
            "gpu_load": 99,
            "gpu_memory_used": 123,
            "gpu_pstate": "P2"}

        msg = String(json.dumps(ipc_load))
        collector._cb(msg)
        loads = collector.get_current_loads(now=now)
        print(loads)



if __name__ == "__main__":
    unittest.main()
