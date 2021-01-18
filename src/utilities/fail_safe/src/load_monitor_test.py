import unittest
from load_monitor import LoadMonitor

class LoadMonitorTest(unittest.TestCase):
    def setUp(self):
        self.monitor = LoadMonitor()

    def test_get_gpu_load(self):
        gpu_load = self.monitor.get_gpu_load()
        self.assertTrue(len(gpu_load) > 0)

    def test_get_cpu_load(self):
        cpu_load = self.monitor.get_cpu_load()
        self.assertTrue(len(cpu_load) > 0)
        self.assertEqual(len(cpu_load.split()), 3)

if __name__ == "__main__":
    unittest.main()
