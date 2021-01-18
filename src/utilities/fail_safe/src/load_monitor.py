"""
Publish cpu/gpu load
"""
import re
import subprocess
import rospy
from std_msgs.msg import String

W_RGX = re.compile(r"load average: (?P<load>[ \.\d,]+)")

def get_hostname():
    """
    Return the hostname of the machine, i.e., /etc/hostname
    """
    return subprocess.check_output(["hostname"]).strip()

def _get_gpu_load_by_nvidia_smi():
    cmd = ["nvidia-smi", "--format=csv,noheader,nounits",
           "--query-gpu=utilization.gpu"]
    return subprocess.check_output(cmd).strip()

def _get_gpu_load_by_tegra_stats():
    # Machines like Xavier do not have nvidia-smi.
    return "NA"

def _get_cpu_load():
    line = subprocess.check_output(["w"]).splitlines()[0]
    match = W_RGX.search(line)
    if match is None:
        return "NA"
    return match.expand(r"\g<load>")


class LoadMonitor(object):
    def __init__(self):
        self.hostname = get_hostname()
        self.use_nvidia_smi = False
        self.check_nvidia_tooling()

    def check_nvidia_tooling(self):
        cmd = ["which", "nvidia-smi"]
        try:
            subprocess.check_output(cmd)
            self.use_nvidia_smi = True
        except subprocess.CalledProcessError:
            self.use_nvidia_smi = False

    def get_gpu_load(self):
        if self.use_nvidia_smi:
            return _get_gpu_load_by_nvidia_smi()
        return _get_gpu_load_by_tegra_stats()

    def get_cpu_load(self):
        return _get_cpu_load()

    def run(self):
        # ROS topic/node name does not allow characters like -
        hostname = self.hostname.replace("-", "_")
        node_name = "LoadMonitor_" + hostname
        rospy.init_node(node_name)
        rospy.logwarn("Init %s", node_name)

        rate = rospy.Rate(1)  # FPS: 1

        cpu_load_topic = "/vehicle/report/{}/cpu_load".format(hostname)
        cpu_load_publisher = rospy.Publisher(cpu_load_topic, String, queue_size=1000)
        rospy.logwarn("Publish data on %s", cpu_load_topic)

        gpu_load_topic = "/vehicle/report/{}/gpu_load".format(hostname)
        gpu_load_publisher = rospy.Publisher(gpu_load_topic, String, queue_size=1000)
        rospy.logwarn("Publish data on %s", gpu_load_topic)
        while not rospy.is_shutdown():
            msg = String()
            msg.data = self.get_cpu_load()
            cpu_load_publisher.publish(msg)
            msg.data = self.get_gpu_load()
            gpu_load_publisher.publish(msg)
            rate.sleep()
