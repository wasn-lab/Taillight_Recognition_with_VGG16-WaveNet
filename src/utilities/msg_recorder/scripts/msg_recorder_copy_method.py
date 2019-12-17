#!/usr/bin/env python
import rospy, rospkg
import math
import time
import sys, os
import signal
import subprocess
import threading
import yaml, json
# File operations
import datetime
# import dircache # <-- Python2.x only, repalce with os.listdir
import shutil
# Args
import argparse
#-------------------------#
try:
    import queue as Queue # Python 3.x
except:
    import Queue # Python 2.x
#-------------------------#
from std_msgs.msg import (
    Empty,
    Bool,
    String,
)

# _rosbag_caller
_rosbag_caller = None

# Publisher
# _recorder_running_pub = None

# For clearing the last line on terminal screen
#---------------------------------------------------#
CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'
def erase_last_lines(n=1, erase=False):
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        if erase:
            sys.stdout.write(ERASE_LINE)
#---------------------------------------------------#

class COPY_QUEUE:
    """
    This is the class for handling the file copying.
    """
    def __init__(self, src_dir, dst_dir, num_copy_thread=3):
        """
        This class is dedicated on doing the following command in an efficient way.
            --> shutil.copy2( (self.src_dir + file_name), self.dst_dir)
        - Prevent duplicated copying
        - Prevent the blocking of copy2() (originally it won't return until the file-copying is done)
            (x) To prevent the traffic jam that slows down the main recording, we just let it block...
        """
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        #
        self.file_Q = Queue.Queue()
        self.copied_file_list = list()
        #
        self.num_copy_thread = num_copy_thread
        self._copy_thread_list = list()
        # Start the polling thread
        self._polling_thread = threading.Thread(target=self._copy_file_listener)
        self._polling_thread.daemon = True # Use daemon to prevent eternal looping
        self._polling_thread.start()


    def add_file(self, file_name):
        """
        This is the public function for entering the name of the file to e copied.
        """
        self.file_Q.put(file_name)

    def _copy_file_worker(self, file_name):
        """
        This is the worker for copying file (blocking function)
        """
        print("[copyQ] Copying <%s>." % file_name)
        shutil.copy2( (self.src_dir + file_name), self.dst_dir)
        print("[copyQ] Finishing copying <%s>." % file_name)

    def _remove_idle_threads(self):
        """
        Remove idle threads
        """
        #--------------------------------#
        _idx = 0
        while _idx < len(self._copy_thread_list):
            if not self._copy_thread_list[_idx].isAlive():
                del self._copy_thread_list[_idx]
                # _idx = 0 # Re-start from beginning...
                # NOTE: the _idx is automatically pointing to the next one
            else:
                _idx += 1
        # print("[CopyQ] Number of thread busying = %d" % len(self._copy_thread_list) )
        #--------------------------------#

    def _copy_file_listener(self):
        """
        This is the thread worker function for listening the file names from queue.
        """
        while True:
            # Note: this thread will only closed if this program is stopped
            while not self.file_Q.empty():
                self._remove_idle_threads()
                if len(self._copy_thread_list) >= self.num_copy_thread:
                    # The pool is full, keep waiting
                    # print("[CopyQ] Copy thread pool is full, keep waiting.")
                    break
                a_file = self.file_Q.get()
                print("[copyQ] Get <%s> from list." % a_file)
                if not a_file in self.copied_file_list:
                    # The file has not been processed
                    self.copied_file_list.append(a_file)
                    # Really copy a file (blocked until finished)
                    # print("[copyQ] Copying <%s>." % a_file)
                    # shutil.copy2( (self.src_dir + a_file), self.dst_dir)
                    _t = threading.Thread(target=self._copy_file_worker, args=(a_file,) )
                    self._copy_thread_list.append(_t)
                    _t.start()
                else:
                    print("[copyQ] Not to copy <%s>." % a_file)
                    # The file is already in the list, not doing copying
                    pass
            #
            if len(self._copy_thread_list) > 0:
                self._remove_idle_threads()
                print("[CopyQ] Number of thread busying = %d" % len(self._copy_thread_list) )
            #
            time.sleep(0.2)


#---------------------------------------------------#
class ROSBAG_CALLER:
    """
    This is the function for handling the rosbag subprocess.
    """
    # Public methods
    def __init__(self, param_dict, node_name="msg_recorder"):
        """
        The param_dict contains the following elements:
        - output_dir_tmp (default: "./rosbag_tmp"): The directry of the output bag files, temporary stored
        - output_dir_kept (default: "./rosbag_backup"): The directry of the output bag files, palced for backuped files
        - bag_name_prefix (default: "record"): The prefix for the generated bag files
        - is_splitting (default: False): Determine if the output bag should be split (once reach the size limit or duration limit)
        - split_size (default: None/1024 MB(when record_duration is None) ): The size to split the file
        - max_split_num (default: None): Maximum number of split files. The oldest file will be delete When the one is generated.
        - record_duration (default: None): 30 (30 sec.), 1m (1 min.), 1h (1 hr.), the duration for recording before stopping/splitting
        - is_compressed (default: false): raw data or compressed data to store (the file all ends with .bag, to check if it's compressed, use rosbag info)
        - compression_method (default: lz4): lz4/bz2; bz2: slow, high compression; lz4: fast, low compression
        - buffsize (default: None <- 256MB): Input buffer size for all messages before bagging, empty -> None -> 256MB
        - chunksize (default: None <- 768KB): Advanced, memory to chunks of SIZE KB before writing to disk. Tune if needed.
        - is_recording_all_topics (default: True): To record all topics or not
        - topics (default: []): a list of topics names (with or without "/" are allowabled)
        #
        - time_pre_trigger (default: 60.0 sec.): Keep all records since time_pre_trigger
        - time_post_trigger (default: 5.0 sec.): Keep all records before time_post_trigger
        """
        # Variables
        self._thread_rosbag = None
        self._ps = None
        self.rosbag_node_name_suffix = "rosbag_subprocess"
        self.rosbag_node_name = node_name + "_"+ self.rosbag_node_name_suffix
        print("rosbag_node_name = %s" % self.rosbag_node_name)
        self._state_sender_func = None
        self._report_sender_func = None
        #
        self._last_trigger_timestamp = 0.0

        # Parameters for rosbag record with default values
        #--------------------------------------------------------------------------#
        self.output_dir_tmp = param_dict.get('output_dir_tmp', "./rosbag_tmp")
        self.output_dir_kept = param_dict.get('output_dir_kept', "./rosbag_backup")
        self.bag_name_prefix = param_dict.get('bag_name_prefix', "record")
        self.is_splitting = param_dict.get('is_splitting', False)
        self.split_size = param_dict.get('split_size', None)
        self.max_split_num = param_dict.get('max_split_num', None)
        self.record_duration = param_dict.get('record_duration', None)
        self.is_compressed = param_dict.get('is_compressed', False)
        self.compression_method = param_dict.get('compression_method', "lz4")
        self.buffsize = param_dict.get('buffsize', None)
        self.chunksize = param_dict.get('chunksize', None)
        self.is_recording_all_topics = param_dict.get('is_recording_all_topics', True)
        self.topic_list = param_dict.get('topics', [])
        #
        self.time_pre_trigger = param_dict.get('time_pre_trigger', 60.0)
        self.time_post_trigger = param_dict.get('time_post_trigger', 5.0)

        # Add '/' at the end
        if self.output_dir_tmp[-1] != "/":
            self.output_dir_tmp += "/"
        if self.output_dir_kept[-1] != "/":
            self.output_dir_kept += "/"

        # The self.output_dir_tmp and self.output_dir_kept cannot be the same.
        if self.output_dir_tmp == self.output_dir_kept:
            self.output_dir_kept += "rosbag_backup/"

        # If both split_size and record_duration are not specified, set the split size
        if self.is_splitting and self.record_duration is None:
            if (self.split_size is None) or (int(self.split_size) <= 0):
                self.split_size = 1024
                print("self.split_size is forced to <%d MB> since both split_size and record_duration are not specified." % int(self.split_size))
        #--------------------------------------------------------------------------#


        # test
        print("self.record_duration = %s" % str(self.record_duration))


        # Preprocessing for parameters
        self.output_dir_tmp = os.path.expandvars( os.path.expanduser(self.output_dir_tmp) )
        print("self.output_dir_tmp = %s" % self.output_dir_tmp)
        self.output_dir_kept = os.path.expandvars( os.path.expanduser(self.output_dir_kept) )
        print("self.output_dir_tmp = %s" % self.output_dir_kept)


        # Creating directories
        try:
            _out = subprocess.check_output(["mkdir", "-p", self.output_dir_tmp], stderr=subprocess.STDOUT)
            print("The directory <%s> has been created." % self.output_dir_tmp)
        except:
            print("The directry <%s> already exists." % self.output_dir_tmp)
            pass

        try:
            _out = subprocess.check_output(["mkdir", "-p", self.output_dir_kept], stderr=subprocess.STDOUT)
            print("The directory <%s> has been created." % self.output_dir_kept)
        except:
            print("The directry <%s> already exists." % self.output_dir_kept)
            pass

        # Initialize the COPY_QUEUE
        self.copyQ = COPY_QUEUE(self.output_dir_tmp, self.output_dir_kept)

        # File list watcher
        self.file_hist_list = list()
        self._file_list_thread = threading.Thread(target=self._file_list_watcher)
        self._file_list_thread.daemon = True # Use daemon to prevent eternal looping
        self._file_list_thread.start()


    def attach_state_sender(self, sender_func):
        """
        This is a public method for binding a "publisher.publish" method
        input:
        - sender_func: a publisher.publish method
        """
        self._state_sender_func = sender_func

    def attach_report_sender(self, sender_func):
        """
        This is a public method for binding a "publisher.publish" method
        input:
        - sender_func: a publisher.publish method
        """
        self._report_sender_func = sender_func

    def start(self, _warning=False):
        """
        To start recording.
        """
        if not self._is_thread_rosbag_valid():
            self._thread_rosbag = threading.Thread(target=self._rosbag_watcher)
            self._thread_rosbag.start()
            return True
        else:
            if _warning:
                print("rosbag is already running, no action")
            return False

    def stop(self, _warning=False):
        """
        To stop recording

        Note: If the stop() is called right after start(), then the rosnode kill might fail (the node is not intialized yet.)
        """
        if self._is_thread_rosbag_valid() and ( (not self._ps is None) and self._ps.poll() is None): # subprocess is running
            return self._terminate_rosbag(_warning=_warning)
        else:
            if _warning:
                print("rosbag is not running, no action.")
            return False

    def split(self, _warning=False):
        """
        To stop a record and start a new one
        """
        self.stop(_warning)
        _count = 0
        _count_max = 1000
        while self._is_thread_rosbag_valid() and (_count < _count_max):
            print("Restart waiting count: %d" % _count)
            _count += 1
            time.sleep(0.1)
        if (_count >= _count_max):
            return False
        else:
            return self.start(_warning)

    def backup(self):
        """
        Backup all the files interset with time zone.

        Use a deamon thread to complete the work even if the program is killed.
        """
        _t = threading.Thread(target=self._keep_files_before_and_after)
        # _t.daemon = True
        _t.start()

    # Private methods
    def _is_thread_rosbag_valid(self):
        """
        Check if the thread is running.
        """
        try:
            return ( (not self._thread_rosbag is None) and self._thread_rosbag.is_alive() )
        except:
            return False

    def _send_rosbag_state(self, is_running, _warning=False):
        """
        This is the output funtion for sending the state of the rosbag out.
        """
        try:
            self._state_sender_func(is_running)
        except:
            if _warning:
                print("No _stat_sender_func or the attached funtion gots error.")

    def _report_event(self, _report_str, _warning=False):
        """
        This is the output funtion for sending the report of the trigger event.
        """
        try:
            msg = String()
            msg.data = _report_str
            self._report_sender_func( msg )
        except:
            if _warning:
                print("No _stat_sender_func or the attached funtion gots error.")



    # Subprocess controll
    #-------------------------------------#
    def _open_rosbag(self):
        """
        The wraper for starting rosbag
        """
        # The program to be run
        #----------------------------#
        """
        # Note; changing working directry here has no effect on the other subprocess calls.
        subprocess.call("cd " + self.output_dir_tmp, shell=True)
        subprocess.call("pwd", shell=True)
        """
        # New subprocess
        #----------------------------#
        # The command
        cmd_list = ["rosbag", "record", ("__name:=%s" % self.rosbag_node_name)]
        # File name prefix
        cmd_list += ["-o", self.bag_name_prefix]
        # Splitting
        if self.is_splitting:
            cmd_list += ["--split"]
            if not self.split_size is None:
                cmd_list += ["--size=%d" % int(self.split_size)]
            if not self.max_split_num is None:
                cmd_list += ["--max-splits=%d" % int(self.max_split_num)]
        # Duration
        if not self.record_duration is None:
            cmd_list += ["--duration=%s" % self.record_duration]
        # Compression
        if self.is_compressed:
            cmd_list += ["--%s" % self.compression_method]
        # self.buffsize
        if not self.buffsize is None:
            cmd_list += ["--buffsize=%d" % int(self.buffsize)]
        # Memory chunksize
        if not self.chunksize is None:
            cmd_list += ["--chunksize=%d" % int(self.chunksize)]
        # topics
        if self.is_recording_all_topics:
            cmd_list += ["-a"]
        else:
            cmd_list += self.topic_list

        #
        print("")
        print("Executing command: %s" % cmd_list)
        print("Working directry: %s" % self.output_dir_tmp)
        print("")
        # self._ps = subprocess.Popen(["rosbag", "record", "__name:=rosbag_subprocess", "-a"]) # rosbag
        # self._ps = subprocess.Popen(["rosbag", "record", ("__name:=%s" % self.rosbag_node_name), "-a"]) # rosbag
        # self.Ps = subprocess.Popen("rosbag record __name:=rosbag_subprocess -a", shell=True) # rosbag
        self._ps = subprocess.Popen(cmd_list, cwd=self.output_dir_tmp)
        #----------------------------#
        return True

    def _terminate_rosbag(self, _warning=False):
        """
        The wraper for killing rosbag
        """
        # First try
        try:
            # self._ps.terminate() # TODO: replace this to a signal to the thread
            # self._ps.send_signal(signal.SIGTERM) # <-- This method cannot cleanly kill the rosbag.
            # subprocess.Popen(["rosnode", "kill", "/rosbag_subprocess"]) # <-- this method can close the rosbag cleanly.
            # subprocess.Popen(["rosnode", "kill", ("/%s" % self.rosbag_node_name) ]) # <-- this method can close the rosbag cleanly.
            subprocess.Popen("rosnode kill /%s" % self.rosbag_node_name, shell=True) # <-- this method can close the rosbag cleanly.
            # subprocess.call(["rosnode", "kill", ("/%s" % self.rosbag_node_name) ]) # <-- this method can close the rosbag cleanly.
            # subprocess.call("rosnode kill /%s" % self.rosbag_node_name, shell=True) # <-- this method can close the rosbag cleanly.
            return True
        except:
            if _warning:
                print("The process cannot be killed by rosnode kill.")
        # Second try
        try:
            self._ps.terminate() #
            if _warning:
                print("The rosbag is killed through SIGTERM, the bag might still be active.")
            return True
        except:
            print("Something wrong while killing the rosbag subprocess")
        #
        return False
    #-------------------------------------#

    def _rosbag_watcher(self):
        """
        This function run as a thread to look after the rosbag process.
        """
        # global _recorder_running_pub
        # The private method to start the process
        self._open_rosbag()
        print("=== Subprocess started.===")
        self._send_rosbag_state(True)
        # _recorder_running_pub.publish(True)
        #
        time_start = time.time()
        while self._ps.poll() is None:
            duration = time.time() - time_start
            # print("                                                  |")
            print("---Subprocess is running, duration = %f     |" % duration)
            # print("                                                  |")
            # erase_last_lines(n=3, erase=False) # To make the cursor position back to the front of line
            time.sleep(1.0)
        result = self._ps.poll()
        print("result = %s" % str(result))
        print("=== Subprocess finished.===")
        self._send_rosbag_state(False)
        # _recorder_running_pub.publish(False)

        # Clear the handle, indicating that no process is running
        # self._ps = None
        return



    # Backing up files
    #----------------------------------------------#
    def _file_list_watcher(self):
        """
        This is the thread worker for listening the file list.
        """
        while True:
            file_list = os.listdir(self.output_dir_tmp)
            file_list.sort()
            for i in range(len(file_list)):
                if file_list[-1-i][-4:] != '.bag':
                    # active file or other file type
                    continue
                # Note that if self.bag_name_prefix is '', then the following is bypassed
                if file_list[-1-i][:len(self.bag_name_prefix)] != self.bag_name_prefix:
                    # Not our bag
                    continue
                if not file_list[-1-i] in self.file_hist_list:
                    self.file_hist_list.append( file_list[-1-i] )
            time.sleep(0.2)

    def _get_latest_inactive_bag(self, timestamp=None):
        """
        This is a helper funtion for finding the latest (inactive) bag file.
        """
        # # file_list = dircache.listdir(self.output_dir_tmp) # Python 2.x only
        # file_list = os.listdir(self.output_dir_tmp)
        # file_list.sort() # Sort in ascending order
        file_list = sorted(self.file_hist_list)
        #
        if timestamp is None:
            target_date = datetime.datetime.now()
        else:
            target_date = datetime.datetime.fromtimestamp(timestamp)
        target_date_formate = target_date.strftime("%Y-%m-%d-%H-%M-%S")
        # print('target_date = %s' % str(target_date))
        # print('target_date_formate = %s' % target_date_formate)
        if self.bag_name_prefix == "":
            target_name_prefix_date = target_date_formate
        else:
            target_name_prefix_date = self.bag_name_prefix + '_' + target_date_formate
        # print('target_name_prefix_date = %s' % target_name_prefix_date)
        # Seraching
        closest_file_name = None
        is_last = True
        # Assume the file_list is sorted in ascending order
        for i in range(len(file_list)):
            # if file_list[-1-i].rfind('.active') >= 0:
            if file_list[-1-i][-4:] != '.bag':
                # active file or other file type
                continue
            # Note that if self.bag_name_prefix is '', then the following is bypassed
            if file_list[-1-i][:len(self.bag_name_prefix)] != self.bag_name_prefix:
                # Not our bag
                continue
            if file_list[-1-i] < target_name_prefix_date:
                closest_file_name = file_list[-1-i]
                break
            else:
                is_last = False
        """
        # Assume the file_list is not sorted
        for i in range(len(file_list)):
            # if file_list[-1-i].rfind('.active') >= 0:
            if file_list[i][-4:] != '.bag':
                # active file or other file type
                continue
            if file_list[i][:len(self.bag_name_prefix)] != self.bag_name_prefix:
                # Not our bag
                continue
            if file_list[i] < target_name_prefix_date:
                if (closest_file_name is None) or (file_list[i] > closest_file_name):
                    # Note: None is actually smaller than anything
                    closest_file_name = file_list[i]
            else:
                is_last = False
            #
        """
        # Note: it's possible to return a None when there is no file in the directory or no inactive file before the given time
        # e.g. We are just recording the first bag (which is acive)
        return (closest_file_name, is_last)

    def _get_list_of_inactive_bag_in_timezone(self, timestamp_start, timestamp_end=None):
        """
        This is a helper funtion for finding the latest (inactive) bag file.
        """
        # # file_list = dircache.listdir(self.output_dir_tmp) # Python 2.x only
        # file_list = os.listdir(self.output_dir_tmp)
        # file_list.sort() # Sort in ascending order
        file_list = sorted(self.file_hist_list)
        #
        target_date_start = datetime.datetime.fromtimestamp(timestamp_start)
        if timestamp_end is None:
            target_date_end = datetime.datetime.now()
        else:
            target_date_end = datetime.datetime.fromtimestamp(timestamp_end)
        target_date_start_formate = target_date_start.strftime("%Y-%m-%d-%H-%M-%S")
        target_date_end_formate = target_date_end.strftime("%Y-%m-%d-%H-%M-%S")
        # print('target_date_start_formate = %s' % target_date_start_formate)
        # print('target_date_end_formate = %s' % target_date_end_formate)
        if self.bag_name_prefix == "":
            target_name_prefix_date_start = target_date_start_formate
            target_name_prefix_date_end = target_date_end_formate
        else:
            target_name_prefix_date_start = self.bag_name_prefix + '_' + target_date_start_formate
            target_name_prefix_date_end = self.bag_name_prefix + '_' + target_date_end_formate
        # print('target_name_prefix_date_start = %s' % target_name_prefix_date_start)
        # print('target_name_prefix_date_end = %s' % target_name_prefix_date_end)
        # Seraching
        file_in_zone_list = []
        # Assume the file_list is sorted in ascending order
        for i in range(len(file_list)):
            # if file_list[-1-i].rfind('.active') >= 0:
            if file_list[-1-i][-4:] != '.bag':
                # active file or other file type
                continue
            # Note that if self.bag_name_prefix is '', then the following is bypassed
            if file_list[-1-i][:len(self.bag_name_prefix)] != self.bag_name_prefix:
                # Not our bag
                continue
            if file_list[-1-i] < target_name_prefix_date_end:
                file_in_zone_list.append(file_list[-1-i])
                if file_list[-1-i] < target_name_prefix_date_start:
                    break
        # Note: it's possible to return an empty list when there is no file in the directory or no inactive file fall in the given time
        # e.g. We are just recording the first bag (which is acive)
        return file_in_zone_list


    def _keep_files_before_and_after(self):
        """
        To keep 2 files: before and after

        The file might look like the following when this function called:
        a.bag
        b.bag.active

        Solution (rough concept):
        1. Backup (copy) the a.bag immediately <-- This is already done in a deamon thread according to the way of this function call. (in case that the main program being closed)
        2. Start another thread (deamon, in case that the main program being closed) for listening that if the b.bag.active has become the b.bag

        Since the original rosbag using date as file name, we don't bother to track the file midification time.
        """
        # Get the current time
        _trigger_timestamp = time.time()
        self._last_trigger_timestamp = _trigger_timestamp
        #
        _pre_trigger_timestamp = _trigger_timestamp - self.time_pre_trigger
        _post_trigger_timestamp = _trigger_timestamp + self.time_post_trigger

        # Find all the "a.bag" files
        file_in_pre_zone_list = self._get_list_of_inactive_bag_in_timezone( _pre_trigger_timestamp, _trigger_timestamp)
        print("file_in_pre_zone_list = %s" % file_in_pre_zone_list)
        # Bacuk up "a.bag", note tha empty list is allowed
        for _F in file_in_pre_zone_list:
            # shutil.copy2( (self.output_dir_tmp + _F), self.output_dir_kept)
            self.copyQ.add_file(_F)

        """
        # Start a deamon thread for watching the "b.bag"
        _t = threading.Thread(target=self._keep_files_after, args=(_trigger_timestamp, _post_trigger_timestamp, file_in_pre_zone_list))
        _t.daemon = True
        _t.start()
        """

        print("===Pre-triggered-file backup finished.===")

        """
    def _keep_files_after(self, _trigger_timestamp, _post_trigger_timestamp, file_in_pre_zone_list):

        # This is a worker for listening the post-triggered bags.

        # Wait ntil reached the _post_trigger_timestamp
        # Note: this is not good for prone to lost the latest post-file.
        # time.sleep(_post_trigger_timestamp - _trigger_timestamp)
        """

        # Start listening, first stage (before _post_trigger_timestamp) and second stage (after _post_trigger_timestamp)
        time_start = time.time()
        while self._is_thread_rosbag_valid():
            duration = time.time() - time_start
            # print("                                                  |")
            # print("                                                  |")
            print("---===Post-triggered file backup thread is running, duration = %f            |" % duration)
            # erase_last_lines(n=3, erase=False) # To make the cursor position back to the front of line
            #
            (closest_file_name, is_last) = self._get_latest_inactive_bag(_post_trigger_timestamp)
            if (not closest_file_name is None) and not closest_file_name in file_in_pre_zone_list:
                # shutil.copy2( (self.output_dir_tmp + closest_file_name), self.output_dir_kept)
                self.copyQ.add_file(closest_file_name)
                file_in_pre_zone_list.append(closest_file_name)
            if not is_last:
                break
            time.sleep(1.0)
        #
        # Find all the rest "b.bag" files (prevent the leak)
        # Note: most of them had been backuped
        file_in_post_zone_list = self._get_list_of_inactive_bag_in_timezone( _trigger_timestamp, _post_trigger_timestamp)
        print("file_in_post_zone_list = %s" % file_in_post_zone_list)
        # Bacuk up "a.bag", note tha empty list is allowed
        for _F in file_in_post_zone_list:
            if not _F in file_in_pre_zone_list:
                file_in_pre_zone_list.append(_F)
                # shutil.copy2( (self.output_dir_tmp + _F), self.output_dir_kept)
                self.copyQ.add_file(_F)

        # Write an indication text
        file_in_pre_zone_list.sort()

        triggered_datetime = datetime.datetime.fromtimestamp(_trigger_timestamp)
        # triggered_datetime_s = target_date.strftime("%Y-%m-%d-%H-%M-%S")
        event_str = "\n\n# Triggered at [%s]\n## backup-files:\n" % str(triggered_datetime)
        for _F in file_in_pre_zone_list:
            event_str += " - %s\n" % _F
        event_str += "\n"

        _fh = open(self.output_dir_kept + "backup_history.txt", "a")
        # _fh.write("\n\n# Triggered at [%s]\n## backup-files:\n" % str(triggered_datetime) )
        # # _fh.write(str(file_in_pre_zone_list))
        # for _F in file_in_pre_zone_list:
        #     _fh.write(" - %s\n" % _F )
        # _fh.write("\n")
        _fh.write( event_str )
        _fh.close()
        #
        # Report by ROS topic
        self._report_event( event_str )
        #
        print("\n===\n\tPost-triggered file backup finished, end of thread.\n===\n")
    #----------------------------------------------#



def _record_cmd_callback(data):
    """
    The callback function for operation command.
    """
    global _rosbag_caller
    if data.data:
        _rosbag_caller.start(_warning=True)
    else:
        _rosbag_caller.stop(_warning=True)

def _backup_trigger_callback(data):
    """
    The callback function for operation command.
    """
    global _rosbag_caller
    _rosbag_caller.backup()




def main(sys_args):
    global _rosbag_caller
    global txt_input
    # global _recorder_running_pub

    # Fix the Python 2.x and 3.x compatibility problem
    #----------------------------------------------------------#
    # The original "input" is deprecated in Python 3.x,
    # and the "raw_input" was renamed to "input"
    # --> For Python 2.x, use "input" as "raw_input"
    # --> For Python 3.x, use "input" normally
    try:
        txt_input = raw_input
    except NameError:
        txt_input = input
    #----------------------------------------------------------#

    # Process arguments
    parser = argparse.ArgumentParser(description="Record ROS messages to rosbag files with enhanced controllability.\nThere are mainly two usage:\n- Manual-mode: simple start/stop record control\n- Auto-mode: Continuous recording with files backup via triggers.")
    #---------------------------#
    # Explicitly chose to auto-mode or manual-mode (exculsive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-A", "--auto", action="store_true", help="continuous recording with triggered backup")
    group.add_argument("-M", "--manual", action="store_true", help="manually control the start/stop of recorder")
    # UI setting
    parser.add_argument("--NO_KEY_IN", action="store_true", help="disable the stdin (keyboard) user-input")
    # Full setting, the following setting will overwrite the above settings.
    parser.add_argument("-d", "--PARAM_DIR", help="specify the directory of the setting-file and topics-file")
    parser.add_argument("-s", "--SETTING_F", help="specify the filename of the setting-file")
    parser.add_argument("-t", "--TOPICS_F", help="specify the filename of the topics-file")
    #---------------------------#
    # _args = parser.parse_args()
    _args, _unknown = parser.parse_known_args()



    #
    rospy.init_node('msg_recorder', anonymous=True)
    #
    _node_name = rospy.get_name()[1:] # Removing the '/'
    print("_node_name = %s" % _node_name)
    #
    rospack = rospkg.RosPack()
    _pack_path = rospack.get_path('msg_recorder')
    print("_pack_path = %s" % _pack_path)
    # Loading parameters
    #---------------------------------------------#
    rospack = rospkg.RosPack()
    pack_path = rospack.get_path('msg_recorder')
    f_path = _pack_path + "/params/"

    # Manual mode
    f_name_params = "rosbag_setting.yaml"
    f_name_topics = "record_topics.txt"


    # Overwriting default values
    #-----------------------------------#
    # Auto/manual
    if _args.auto:
        f_name_params = "rosbag_setting_auto.yaml"
        f_name_topics = "record_topics_auto.txt"
    elif _args.manual:
        # Default is manual-mode, nothing to do
        pass

    # Customize
    if not _args.PARAM_DIR is None:
        f_path = _args.PARAM_DIR
        if f_path[-1] != '/':
            f_path += '/'
    if not _args.SETTING_F is None:
        f_name_params = _args.SETTING_F
    if not _args.TOPICS_F is None:
        f_name_topics = _args.TOPICS_F
    #-----------------------------------#

    # Read param file
    #------------------------#
    _f = open( (f_path+f_name_params),'r')
    params_raw = _f.read()
    _f.close()
    param_dict = yaml.load(params_raw)
    #------------------------#

    # Read topic_list file
    #------------------------#
    topic_list = []
    _f = open( (f_path+f_name_topics),'r')
    for _s in _f:
        # Remove the space and '\n'
        _s1 = _s.rstrip().lstrip()
        # Deal with coments
        _idx_comment = _s1.find('#')
        if _idx_comment >= 0: # Do find a '#'
            _s1 = _s1[:_idx_comment].rstrip() # Remove the comment parts
        if len(_s1) > 0: # Append non-empty string (after stripping)
            topic_list.append(_s1)
    _f.close()
    #------------------------#


    # Print the params
    # print("param_dict = %s" % str(param_dict))
    print("\nsettings (in json format):\n%s" % json.dumps(param_dict, indent=4))
    print("\n\ntopic_list:\n---------------" )
    for _tp in topic_list:
        print(_tp)
    print("---------------\n\n" )


    # Add the 'topics' to param_dict
    param_dict['topics'] = topic_list


    # test, the param_dict after combination
    # print("param_dict = %s" % str(param_dict))
    # print("param_dict (in json format):\n%s" % json.dumps(param_dict, indent=4))
    #---------------------------------------------#

    # Init ROS communication interface
    #--------------------------------------#
    # Subscriber
    rospy.Subscriber("/REC/record", Bool, _record_cmd_callback)
    rospy.Subscriber("/REC/req_backup", Empty, _backup_trigger_callback)
    # Publisher
    _recorder_running_pub = rospy.Publisher("/REC/is_recording", Bool, queue_size=10, latch=True) #
    _recorder_running_pub.publish(False)
    _trigger_event_report_pub = rospy.Publisher("/REC/trigger_report", String, queue_size=20, latch=True) #
    #--------------------------------------#





    # The manager for rosbag record
    #---------------------------------------------#
    _rosbag_caller = ROSBAG_CALLER(param_dict, _node_name)
    _rosbag_caller.attach_state_sender(_recorder_running_pub.publish)
    _rosbag_caller.attach_report_sender(_trigger_event_report_pub.publish)

    # Start at beginning
    if param_dict['start_at_begining']:
        _rosbag_caller.start(_warning=True)
    #---------------------------------------------#



    # Determine if we are using keyboard input
    _is_key_in = _args.NO_KEY_IN


    # Loop for user command via stdin
    while not rospy.is_shutdown():
        if not _is_key_in:
            # A blocking std_in function
            # str_in = raw_input("\n----------------------\nType a command and press ENTER:\n----------------------\ns:start \nt:terminate \nc:cut file \nk:keep file \nq:quit \n----------------------\n>>> ") # <-- This can only be used in Python 2.x
            str_in = txt_input("\n----------------------\nType a command and press ENTER:\n----------------------\ns:start \nt:terminate \nc:cut file \nk:keep file \nq:quit \n----------------------\n>>> ") # The wrapper is called at the beginning of the main()
            #
            if str_in == 's': # Start
                _rosbag_caller.start(_warning=True)
            elif str_in == 't': # Terminate
                _rosbag_caller.stop(_warning=True)
            elif str_in == 'c': # Cut
                _rosbag_caller.split(_warning=True)
            elif str_in == 'k': # Keep
                _rosbag_caller.backup()
            elif str_in == 'q': # Quit
                _rosbag_caller.stop(_warning=False)
                break
            else:
                pass
        #
        time.sleep(0.5)
    print("End of main loop.")




if __name__ == '__main__':

    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
    print("End of recorder.")
