# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import datetime
import unittest
from load_collector import LoadCollector

class LoadCollectorTest(unittest.TestCase):
    def test_get_current_loads(self):
        now = datetime.datetime.now()

        collector = LoadCollector()
        collector.ipcs = ["my"]
        collector.setup_records()
        collector.records["my"]["cpu_load"] = "0.16, 0.21, 0.18"
        collector.timestamps["my"]["cpu_load"] = now
        collector.records["my"]["gpu_load"] = "47"
        collector.timestamps["my"]["gpu_load"] = now

        delta = datetime.timedelta(seconds=1)
        for _ in range(3):
            collector.timestamps["my"]["cpu_load"] -= delta
            collector.timestamps["my"]["gpu_load"] -= delta
            loads = collector.get_current_loads(now=now)
            self.assertEqual(loads["my"]["cpu_load"], "0.16, 0.21, 0.18")
            self.assertEqual(loads["my"]["gpu_load"], "47")
        collector.timestamps["my"]["cpu_load"] -= delta
        collector.timestamps["my"]["gpu_load"] -= delta
        loads = collector.get_current_loads(now=now)
        self.assertEqual(loads["my"]["cpu_load"], "NA")
        self.assertEqual(loads["my"]["gpu_load"], "NA")


if __name__ == "__main__":
    unittest.main()
