import unittest
from load_monitor import get_hostname, LoadMonitor

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

    def test_get_hostname(self):
        name = get_hostname()
        self.assertTrue(len(name) > 0)
        self.assertFalse(name[-1].isspace())

if __name__ == "__main__":
    unittest.main()
