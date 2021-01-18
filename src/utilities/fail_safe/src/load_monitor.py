import argparse
import re
import time
import rospy
import subprocess

W_RGX = re.compile(r"load average: (?P<load>[ \.\d,]+)")

class LoadMonitor(object):
    def __init__(self):
        self.hostname = self.__get_hostname()
        self.use_nvidia_smi = False
        self.check_nvidia_tooling()

    def check_nvidia_tooling(self):
        cmd = ["which", "nvidia-smi"]
        try:
            ret = subprocess.check_output(cmd)
            self.use_nvidia_smi = True
        except subprocess.CalledProcessError:
            self.use_nvidia_smi = False

    def get_gpu_load(self):
        if self.use_nvidia_smi:
            return self.__get_gpu_load_by_nvidia_smi()
        return self.__get_gpu_load_by_tegra_stats()

    def __get_gpu_load_by_nvidia_smi(self):
        cmd = ["nvidia-smi", "--format=csv,noheader,nounits",
               "--query-gpu=utilization.gpu"]
        return subprocess.check_output(cmd).strip()

    def __get_gpu_load_by_tegra_stats(self):
        # Machines like Xavier do not have nvidia-smi.
        return "NA"

    def get_cpu_load(self):
        line = subprocess.check_output(["w"]).splitlines()[0]
        match = W_RGX.search(line)
        return match.expand(r"\g<load>")

    def __get_hostname(self):
        return subprocess.check_output(["hostname"]).strip()
