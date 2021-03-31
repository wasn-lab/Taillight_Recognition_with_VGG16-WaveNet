# Copyright (c) 2021, Industrial Technology and Research Institute.
# All rights reserved.
import unittest
import json
from load_monitor import (get_hostname, LoadMonitor, _parse_nvidia_smi_output,
                          get_nproc, _parse_cpu_load_output)

class LoadMonitorTest(unittest.TestCase):
    def setUp(self):
        self.monitor = LoadMonitor()

    def test_get_hostname(self):
        name = get_hostname()
        self.assertTrue(len(name) > 0)
        self.assertFalse(name[-1].isspace())

    def test_get_nproc_as_str(self):
        nproc = get_nproc()
        self.assertTrue(nproc >= 4)  # We don't have low-end computers.

    def test_get_ipc_load(self):
        ipc_load = self.monitor.get_ipc_load()
        jdata = json.loads(ipc_load)
        # print(jdata)
        self.assertEqual(jdata["hostname"], get_hostname())
        self.assertTrue(isinstance(jdata["cpu_load_threshold"], float))
        self.assertTrue(isinstance(jdata["cpu_load"], float))
        self.assertTrue(isinstance(jdata["gpu_load"], float))

    def test_parse_cpu_load_output(self):
        text = " 10:48:40 up  2:04,  1 user,  load average: 0.34, 0.19, 0.17"
        res = _parse_cpu_load_output(text)
        self.assertEqual(res, "0.17")

        text = " 10:57:05 up 18:43,  9 users,  load average: 13.18, 13.21, 10.15"
        res = _parse_cpu_load_output(text)
        self.assertEqual(res, "10.15")

    def test_get_gpu_load_fallback(self):
        line = "| 22%   34C    P8     9W / 250W |     44MiB / 11016MiB |      17%      Default |"
        self.assertEqual(_parse_nvidia_smi_output(line), "17")

        line = "|   0  GeForce RTX 208...  Off  | 00000000:06:00.0 Off |                  N/A |"
        self.assertEqual(_parse_nvidia_smi_output(line), "")


if __name__ == "__main__":
    unittest.main()
